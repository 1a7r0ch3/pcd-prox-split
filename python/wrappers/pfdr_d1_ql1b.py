import numpy as np
import os 
import sys

sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                              "../bin"))

from pfdr_d1_ql1b_cpy import pfdr_d1_ql1b_cpy

def pfdr_d1_ql1b(Y, A, edges, edge_weights=None, Yl1=None, l1_weights=None, 
                 low_bnd=None, upp_bnd=None, L=None, rho=1.,  
                 cond_min=1e-2, dif_rcd=0., dif_tol=None, 
                 it_max=int(1e4), verbose=int(1e3), AtA_if_square=True, 
                 compute_Obj=False, compute_Dif=False):
    
    """
    X, it, Obj, Dif = pfdr_d1_ql1b(
            Y | AtY, A | AtA, edges, edge_weights=None, Yl1=None, 
            l1_weights=None, low_bnd=None, upp_bnd=None, L=None, rho=1.0,
            cond_min=1e-2, dif_rcd=0.0, dif_tol=1e-3*cp_dif_tol,
            it_max=int(1e4), verbose=int(1e3), AtA_if_square=True,
            compute_Obj=False, compute_Dif=False)

    PFDR with d1 (total variation) penalization, with a 
    quadratic functional, l1 penalization and box constraints:

    minimize functional over a graph G = (V, E)

        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||yl1 - x||_l1 + i_[m,M](x)

    where y in R^N, x in R^V, A in R^{N-by-|V|}
          ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
          ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
          and the convex indicator
          i_[m,M] = infinity if it exists v in V such that x_v < m_v or 
          x_v > M_v
                  = 0 otherwise;

    using preconditioned forward-Douglas-Rachford 
    splitting algorithm.

    It is easy to introduce a SDP metric weighting the squared l2-norm
    between y and A x. Indeed, if M is the matrix of such a SDP metric,
    ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
    Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
    Moreover, when A is the identity and M is diagonal (weighted square l2 
    distance between x and y), one should call on the precomposed version 
    (see below) with Y <- DDy = My and A <- D2 = M.

    INPUTS: real numeric type is either float32 or float64, not both;
            indices numeric type is uint32.

    NOTA: by default, components are identified using uint16_t identifiers; 
    this can be easily changed in the wrapper source if more than 65535
    components are expected (recompilation is necessary)

    Y - observations, (real) array of length N (direct matricial case) or of 
        length V (premultiplied to the left by A^t), or empty matrix (for all 
        zeros)
    A - matrix, (real) N-by-V array (direct matricial case), or V-by-V array
        (premultiplied to the left by A^t), or V-by-1 array (square diagonal
        of A^t A = A^2), or nonzero scalar (for identity matrix), or zero 
        scalar (for no quadratic part); if N = V in a direct matricial case, 
        the last argument AtA_if_square must be set to false
    edges - list of edges, (uint32) array of length 2E; 
        edge number e connects vertices indexed at edges(2*e - 1) and 
        edges(2*e); every vertex should belong to at least one edge with a 
        nonzero penalization coefficient. If it is not the case, a workaround 
        is to add an edge from the vertex to itself with a small nonzero weight
    edge_weights - array of length E or scalar for homogeneous weights (real)
    Yl1        - offset for l1 penalty, (real) array of length V,
                 or empty matrix (for all zeros)
    l1_weights - array of length V or scalar for homogeneous weights (real)
                 set to zero for no l1 penalization 
    low_bnd    - array of length V or scalar (real)
                 set to negative infinity for no lower bound
    upp_bnd    - array of length V or scalar (real)
                 set to positive infinity for no upper bound
    L - information on Lipschitzianity of the operator A^* A;
        scalar satisfying 0 < L <= ||A^* A||, or
        diagonal matrix (real array of length V) satisfying
            0 < L and ||L^(-1/2) A^* A L^(-1/2)|| <= 1, or
        empty matrix for automatic estimation 
    rho   - relaxation parameter, 0 < rho < 2
                 1 is a conservative value; 1.5 often speeds up convergence
    cond_min - stability of preconditioning; 0 < cond_min < 1;
                    corresponds roughly the minimum ratio to the maximum 
                    descent metric; 1e-2 is typical; a smaller value might 
                    enhance preconditioning
    dif_rcd - reconditioning criterion on iterate evolution;
                   a reconditioning is performed if relative changes of the
                   iterate drops below dif_rcd
                   warning: reconditioning might temporarily draw minimizer 
                   away from solution, and give bad subproblem solutions
    dif_tol - stopping criterion on iterate evolution; algorithm stops if
                   relative changes (in Euclidean norm) is less than dif_tol
                   1e-3*cp_dif_tol is a conservative value TODO
    it_max  - maximum number of iterations 1e4 iterations provides enough
                   precision for most subproblems
    verbose      - if nonzero, display information on the progress, every
                   'verbose' PFDR iterations
    AtA_if_square - if A is square, set this to false for direct matricial case
    compute_Obj  - compute the objective functional along iterations 
    compute_Dif  - compute relative evolution along iterations 

    OUTPUTS: Obj, Dif are optional outputs, set optional input
        compute_Obj, compute_Dif to True to get them 

    X - TODO
    it - actual number of cut-pursuit iterations performed
    Obj - if requested, the values of the objective functional along iterations
        (array of length it + 1); in the precomputed A^t A version, a
        constant 1/2||Y||^2 in the quadratic part is omited
    Dif  - if requested, the iterate evolution along iterations
        (array of length it)

    Parallel implementation with OpenMP API.

    H. Raguet, A Note on the Forward-Douglas--Rachford Splitting for Monotone 
    Inclusion and Convex Optimization Optimization Letters, 2018, 1-24

    Baudoin Camille 2019
    """

    # Determine the type of float argument (real_t) 
    # real type is determined by the first parameter Y if nonempty; 
    # or by the second parameter A is nonscalar;
    # or by the parameter Yl1
    if type(Yl1) == np.ndarray and Y.size > 0:
        real_t = Y.dtype
    elif type(Yl1) == np.ndarray and Yl1.size > 0:
        real_t = Yl1.dtype
    else:
        raise TypeError(("At least one of arguments 'Y' or 'Yl1' "
                         "must be provided as a nonempty numpy array."))

    if real_t not in ["float32", "float64"]:
        raise TypeError(("Currently, the real numeric type must be float32 or"
                         " float64."))

    # Convert in numpy array scalar entry: Y, A, edges, 
    # edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, and define float numpy 
    # array argument with the right float type, if empty:
    if type(Y) != np.ndarray:
        if Y == None:
            Y = np.array([], dtype=real_t)
        else:
            raise TypeError("Argument 'Y' must be a numpy array.")

    if type(A) != np.ndarray:
        if type(A) == list:
            raise TypeError("Argument 'A' must be a scalar or a numpy array.")
        else:
            A = np.array([A], real_t)

    if type(edges) != np.ndarray or edges.dtype != "uint32":
        raise TypeError(("Argument 'edges' must be a numpy array of "
                         "type uint32."))

    if type(edge_weights) != np.ndarray:
        if type(edge_weights) == list:
            raise TypeError("Argument 'edge_weights' must be a scalar or a "
                            "numpy array.")
        elif edge_weights != None:
            edge_weights = np.array([edge_weights], dtype=real_t)
        else:
            edge_weights = np.array([1.0], dtype=real_t)

    if type(Yl1) != np.ndarray:
        if type(Yl1) == list:
            raise TypeError("Argument 'Yl1' must be a scalar or a numpy "
                            "array.")
        elif Yl1 != None:
            Yl1 = np.array([Yl1], dtype=real_t)
        else:
            Yl1 = np.array([], dtype=real_t)

    if type(l1_weights) != np.ndarray:
        if type(l1_weights) == list:
            raise TypeError("Argument 'l1_weights' must be a scalar or a numpy"
                            " array.")
        elif l1_weights != None:
            l1_weights = np.array([l1_weights], dtype=real_t)
        else:
            l1_weights = np.array([0.0], dtype=real_t)

    if type(low_bnd) != np.ndarray:
        if type(low_bnd) == list:
            raise TypeError("Argument 'low_bnd' must be a scalar or a numpy "
                            "array.")
        elif low_bnd != None:
            low_bnd = np.array([low_bnd], dtype=real_t)
        else: 
            low_bnd = np.array([-np.inf], dtype=real_t)

    if type(upp_bnd) != np.ndarray:
        if type(upp_bnd) == list:
            raise TypeError("Argument 'upp_bnd' must be a scalar or a numpy "
                            "array.")
        elif upp_bnd != None:
            upp_bnd = np.array([upp_bnd], dtype=real_t)
        else: 
            upp_bnd = np.array([np.inf], dtype=real_t)

    if type(L) != np.ndarray:
        if type(L) == list:
            raise TypeError("Argument 'L' must be a scalar or a numpy array ")
        elif L != None:
            L = np.array([L], dtype=real_t)
        else:
            L = np.array([], dtype=real_t)

    # Determine V and check the graph structure
    if A.ndim > 1 and A.shape[1] != 1:
        V = A.shape[1] 
    else if A.shape[0] == 1:
        if Y.size > 0:
            V = Y.size
        else if Yl1.size > 0:
            V = Yl1.size
    else:
        V = A.shape[0]
    
    if py_first_edge.size != (V + 1):
        raise ValueError("Cut-pursuit d1 quadratic l1 bounds: argument "
                         "'first_edge' should contain |V + 1| = {0} elements, "
              "but {1} are given.".format(V+1, first_edge.size))

    # Check type of all numpy.array arguments of type float (Y, A, 
    # edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, L) 
    for name, ar_args in zip(
            ["Y", "A", "edge_weights", "Yl1", "l1_weights", "low_bnd",
             "upp_bnd", "L"],
            [Y, A, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, L]):
        if ar_args.dtype != real_t:
            raise TypeError("argument '{0}' must be of type '{1}'"
                            .format(name, real_t))

    # Check fortran continuity of all multidimensional numpy.array arguments
    if not(Y.flags["F_CONTIGUOUS"]):
        raise TypeError("argument 'Y' must be F_CONTIGUOUS")
    if not(A.flags["F_CONTIGUOUS"]):
        raise TypeError("argument 'A' must be F_CONTIGUOUS")

    # Convert in float64 all float arguments if needed (rho, 
    # cond_min, dif_rcd, dif_tol) 
    rho = float(rho)
    cond_min = float(cond_min)
    dif_rcd = float(dif_rcd)
    dif_tol = float(dif_tol)
     
    # Convert all int arguments (it_max, verbose) in ints: 
    it_max = int(it_max)
    verbose = int(verbose)

    # Check type of all booleen arguments (AtA_if_square, compute_Obj,
    # , compute_Dif)
    for name, b_args in zip(
            ["AtA_if_square", "compute_Obj", "compute_Dif"],
            [AtA_if_square, compute_Obj, compute_Dif]):
        if type(b_args) != bool:
            raise TypeError("argument '{0}' must be boolean".format(name))


    # Call wrapper python in C  
    X, it, Obj, Dif = d1_ql1b_cpy(
            Y, A, edges, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, L, 
            rho, cond_min, dif_rcd, dif_tol, it_max, 
            verbose, AtA_if_square, real_t == "float64", compute_Obj, 
            compute_Dif) 

    it = it[0]
    
    # Return output depending of the optional output needed
    if (compute_Obj and compute_Dif):
        return X, Obj, Dif
    elif (compute_Obj):
        return X, it, Obj
    elif (compute_Dif):
        return X, it, Dif
    else:
        return X, it



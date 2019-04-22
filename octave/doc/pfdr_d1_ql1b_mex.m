function [X, it, Obj, Dif] = pfdr_d1_ql1b_mex(Y, A, edges, d1_weights, ...
    Yl1, l1_weights, low_bnd, upp_bnd, L, rho, cond_min, dif_rcd, dif_tol, ...
    it_max, verbose, AtA_if_square)
%
%        [X, it, Obj, Dif] = pfdr_d1_ql1b_mex(Y | AtY, A | AtA, edges,
%   d1_weights = 1, Yl1 = [], l1_weights = 0, low_bnd = -Inf, upp_bnd = Inf,
%   L = [], rho = 1., cond_min = 1e-2, dif_rcd = 1e-4, dif_tol = 1e-5,
%   it_max = 1e4, verbose = 1e2, AtA_if_square = true)
%
% Minimize functional over a graph G = (V, E)
%
%        F(x) = 1/2 ||y - A x||^2 + ||x||_d1 + ||x||_l1 + i_{[m,M]}
%
% where y in R^N, x in R^V, A in R^{N-by-|V|}
%      ||x||_d1 = sum_{uv in E} w_d1_uv |x_u - x_v|,
%      ||x||_l1 = sum_{v  in V} w_l1_v |x_v|,
% and the convex indicator
%      i_[m,M] = infinity if it exists v in V such that x_v < m_v or x_v > M_v
%              = 0 otherwise;
%
% using preconditioned forward-Douglas-Rachford splitting algorithm.
%
% It is easy to introduce a SDP metric weighting the squared l2-norm
% between y and A x. Indeed, if M is the matrix of such a SDP metric,
%   ||y - A x||_M^2 = ||Dy - D A x||^2, with D = M^(1/2).
% Thus, it is sufficient to call the method with Y <- Dy, and A <- D A.
% Moreover, when A is the identity and M is diagonal (weighted square l2 
% distance between x and y), one should call on the precomposed version 
% (see below) with Y <- DDy = My and A <- D2 = M.
%
%
% INPUTS: real numeric type is either single or double, not both;
%         indices are C-style (start at 0) of type uint32
%         inputs with default arguments can be omited but all the subsequent
%         arguments must then be omited as well
%
% Y - observations, (real) array of length N (direct matricial case), or
%                          array of length V (left-premult. by A^t), or
%                          empty matrix (for all zeros)
% A - matrix, (real) N-by-V array (direct matricial case), or
%                    V-by-V array (premultiplied to the left by A^t), or
%                    V-by-1 array (_square_ diagonal of A^t A = A^2), or
%                    nonzero scalar (for identity matrix), or
%                    zero scalar (for no quadratic part);
%     for an arbitrary scalar matrix, use identity and scale observations and
%     penalizations accordingly
%     if N = V in a direct matricial case, the last argument 'AtA_if_square'
%     must be set to false
% edges - list of edges (C-style indices), (uint32) array of length 2E;
%     edge number e connects vertices indexed at edges(2*e - 1) and edges(2*e);
%     every vertex should belong to at least one edge with a nonzero 
%     penalization coefficient. If it is not the case, a workaround is to add 
%     an edge from the vertex to itself with a small nonzero weight
% d1_weights - array of length E or scalar for homogeneous weights (real)
% Yl1 - offset for l1 penalty, (real) array of length V, or empty matrix for
%     all zeros
% l1_weights - array of length V or scalar for homogeneous weights (real);
%     set to zero for no l1 penalization 
% low_bnd - array of length V or scalar (real);
%     set to negative infinity for no lower bound
% upp_bnd - array of length V or scalar (real);
%     set to positive infinity for no upper bound
% L - information on Lipschitzianity of the operator A^* A;
%         scalar satisfying 0 < L <= ||A^* A||, or
%         diagonal matrix (real array of length V) satisfying
%             0 < L and ||L^(-1/2) A^* A L^(-1/2)|| <= 1, or
%         empty matrix for automatic estimation 
% rho - relaxation parameter, 0 < rho < 2;
%     1 is a conservative value; 1.5 often speeds up convergence
% cond_min - stability of preconditioning; 0 < cond_min < 1;
%     corresponds roughly to the minimum ratio between different directions of
%     the descent metric; 1e-2 is a typical value;
%     smaller values might enhance preconditioning but might also make it
%     unstable; increase this value if iteration steps seem to get too small
% dif_rcd - reconditioning criterion on iterate evolution;
%     a reconditioning is performed if relative changes of the iterate drops
%     below dif_rcd; it is then divided by 10;
%     10*dif_tol is a typical value, 1e2*dif_tol or 1e3*dif_tol might speed up
%     convergence;
%     WARNING: reconditioning might temporarily draw minimizer away from
%     solution, it is advised to monitor objective value when using
%     reconditioning
% dif_tol - stopping criterion on iterate evolution; algorithm stops if
%     relative changes (in Euclidean norm) is less than dif_tol;
%     1e-5 is a typical value; a lower one can give better precision but with
%     longer computational time
% it_max - maximum number of iterations;
%     usually depends on the size of the problems in relation to the available
%     computational budget
% verbose - if nonzero, display information on the progress, every 'verbose'
%     iterations
%
% OUTPUTS:
%
% X - final minimizer, array of length V (real)
% it - actual number of iterations performed
% Obj - the values of the objective functional along iterations;
%     array of length cp_it + 1;
%     WARNING: in the precomputed A^t A version (including diagonal or identity
%     case), a constant 1/2||Y||^2 in the quadratic part is omited
% Dif  - if requested, the iterate evolution along iterations;
%     array of length it
% 
% Parallel implementation with OpenMP API.
%
% H. Raguet and L. Landrieu, Preconditioning of a Generalized Forward-Backward
% Splitting and Application to Optimization on Graphs, SIAM Journal on Imaging
% Sciences, 2015, 8, 2706-2739
%
% H. Raguet, A Note on the Forward-Douglas-Rachford Splitting for Monotone 
% Inclusion and Convex Optimization, Optimization Letters, 2018, 1-24
%
% Hugo Raguet 2016, 2018, 2019

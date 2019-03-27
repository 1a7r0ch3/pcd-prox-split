/*=============================================================================
 * [X, it, Obj, Dif] = pfdr_d1_ql1b_mex(Y | AtY, A | AtA, edges,
 *      d1_weights = 1.0, Yl1 = [], l1_weights = 0.0, low_bnd = -Inf,
 *      upp_bnd = Inf, L = [], rho = 1.0, cond_min = 1e-2, dif_rcd = 1e-4,
 *      dif_tol = 1e-5, it_max = 1e4, verbose = 1e2, AtA_if_square = true)
 * 
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/pfdr_d1_ql1b.hpp"

using namespace std;

/* vertex_t must be able to represent the numbers of vertices */
typedef uint32_t vertex_t;
# define VERTEX_CLASS mxUINT32_CLASS
# define VERTEX_ID "uint32"

/* arrays with arguments type */
static const int args_real_t[] = {0, 1, 3, 4, 5, 6, 7, 8};
static const int n_real_t = 8;
static const int args_vertex_t[] = {2};
static const int n_vertex_t = 1;

/* function for checking arguments type */
static void check_args(int nrhs, const mxArray *prhs[], const int* args,
    int n, mxClassID id, const char* id_name)
{
    for (int i = 0; i < n; i++){
        if (nrhs > args[i] && mxGetClassID(prhs[args[i]]) != id
            && mxGetNumberOfElements(prhs[args[i]]) > 1){
            mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
                "argument %d is of class %s, but class %s is expected.",
                args[i] + 1, mxGetClassName(prhs[args[i]]), id_name);
        }
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, id, mxREAL);
    if (size){
        mxSetM(row, 1);
        mxSetN(row, size);
        buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
        mxSetData(row, (void*) buffer);
    }else{
        mxFree((void*) buffer);
    }
    return row;
}

/* template for handling both single and double precisions */
template<typename real_t, mxClassID mxREAL_CLASS>
static void pfdr_d1_ql1b_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    /**  get inputs  **/

    /* quadratic functional */
    size_t N = mxGetM(prhs[1]);
    vertex_t V = mxGetN(prhs[1]);

    const real_t *Y = !mxIsEmpty(prhs[0]) ?
        (real_t*) mxGetData(prhs[0]) : nullptr;
    const real_t *A = (N == 1 && V == 1) ?
        nullptr : (real_t*) mxGetData(prhs[1]);
    const real_t a = (N == 1 && V == 1) ?
        mxGetScalar(prhs[1]) : 1.0;

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (!mxIsEmpty(prhs[0])){ /* fidelity is square l2 */
                V = mxGetNumberOfElements(prhs[0]);
            }else if (!mxIsEmpty(prhs[4])){ /* fidelity is only l1 */
                V = mxGetNumberOfElements(prhs[4]);
            }else{ /* should not happen */
                mexErrMsgIdAndTxt("MEX", "PFDR graph d1 quadratic l1 bounds: "
                    "arguments Y and Yl1 cannot be both empty.");
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = DIAG_ATA;
    }else if (V == N && (nrhs < 16 || mxIsLogicalScalarTrue(prhs[15]))){
        N = FULL_ATA; // A and Y are left-premultiplied by A^t
    }

    /* graph structure */
    check_args(nrhs, prhs, args_vertex_t, n_vertex_t, VERTEX_CLASS, VERTEX_ID);
    size_t E = mxGetNumberOfElements(prhs[2])/2;
    const vertex_t *edges = (vertex_t*) mxGetData(prhs[2]);

    /* penalizations */
    const real_t* d1_weights =
        (nrhs > 3 && mxGetNumberOfElements(prhs[3]) > 1) ?
        (real_t*) mxGetData(prhs[3]) : nullptr;
    real_t homo_d1_weight =
        (nrhs > 3 && mxGetNumberOfElements(prhs[3]) == 1) ?
        mxGetScalar(prhs[3]) : 1;

    const real_t* Yl1 = (nrhs > 4 && !mxIsEmpty(prhs[4])) ?
        (real_t*) mxGetData(prhs[4]) : nullptr;
    const real_t* l1_weights =
        (nrhs > 5 && mxGetNumberOfElements(prhs[5]) > 1) ?
        (real_t*) mxGetData(prhs[5]) : nullptr;
    real_t homo_l1_weight =
        (nrhs > 5 && mxGetNumberOfElements(prhs[5]) == 1) ?
        mxGetScalar(prhs[5]) : 0.0;

    const real_t* low_bnd =
        (nrhs > 6 && mxGetNumberOfElements(prhs[6]) > 1) ?
        (real_t*) mxGetData(prhs[6]) : nullptr;
    real_t homo_low_bnd =
        (nrhs > 6 && mxGetNumberOfElements(prhs[6]) == 1) ?
        mxGetScalar(prhs[6]) : -INF_REAL;

    const real_t* upp_bnd =
        (nrhs > 7 && mxGetNumberOfElements(prhs[7]) > 1) ?
        (real_t*) mxGetData(prhs[7]) : nullptr;
    real_t homo_upp_bnd =
        (nrhs > 7 && mxGetNumberOfElements(prhs[7]) == 1) ?
        mxGetScalar(prhs[7]) : INF_REAL;

    const real_t* L =
        (nrhs > 8 && mxGetNumberOfElements(prhs[8]) > 1) ?
        (real_t*) mxGetData(prhs[8]) : nullptr;
    real_t l =
        (nrhs > 8 && mxGetNumberOfElements(prhs[8]) == 1) ?
        mxGetScalar(prhs[8]) : 0.0;

    /* algorithmic parameters */
    real_t rho = (nrhs > 9) ? mxGetScalar(prhs[9]) : 1.0;
    real_t cond_min = (nrhs > 10) ? mxGetScalar(prhs[10]) : 1e-2;
    real_t dif_rcd = (nrhs > 11) ? mxGetScalar(prhs[11]) : 1e-4;
    real_t dif_tol = (nrhs > 12) ? mxGetScalar(prhs[12]) : 1e-5;
    int it_max = (nrhs > 13) ? mxGetScalar(prhs[13]) : 1e4;
    int verbose = (nrhs > 14) ? mxGetScalar(prhs[14]) : 1e2;

    /**  create output  **/

    /* determines real_t class ID */

    plhs[0] = mxCreateNumericMatrix(V, 1, mxREAL_CLASS, mxREAL);
    real_t *X = (real_t*) mxGetPr(plhs[0]);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[1]);

    real_t* Obj = nlhs > 2 ?
        (real_t*) mxMalloc(sizeof(real_t)*(it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(double)*it_max) : nullptr;

    /**  preconditioned forward-Douglas-Rachford  **/

    Pfdr_d1_ql1b<real_t, vertex_t> *pfdr =
        new Pfdr_d1_ql1b<real_t, vertex_t>(V, E, edges);

    pfdr->set_edge_weights(d1_weights, homo_d1_weight);
    pfdr->set_quadratic(Y, N, A, a);
    pfdr->set_l1(l1_weights, homo_l1_weight, Yl1);
    pfdr->set_bounds(low_bnd, homo_low_bnd, upp_bnd, homo_upp_bnd);
    if (!mxIsEmpty(prhs[8])){ pfdr->set_lipschitz_param(L, l); }
    pfdr->set_conditioning_param(cond_min, dif_rcd);
    pfdr->set_relaxation(rho);
    pfdr->set_algo_param(dif_tol, it_max, verbose);
    pfdr->set_monitoring_arrays(Obj, Dif);
    pfdr->set_iterate(X);
    pfdr->initialize_iterate();

    *it = pfdr->precond_proximal_splitting();

    pfdr->set_iterate(nullptr); // prevent X to be free()'d
    delete pfdr;

    /**  resize monitoring arrays and assign to outputs  **/
    if (nlhs > 2){
        plhs[2] = resize_and_create_mxRow(Obj, *it + 1, mxREAL_CLASS);
    }
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Dif, *it, mxREAL_CLASS);
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the first parameter Y if nonempty;
     * or by the second parameter A if nonempty and nonscalar;
     * or by the fifth parameter Yl1 */
    if ((!mxIsEmpty(prhs[0]) && mxIsDouble(prhs[0])) ||
        (mxGetNumberOfElements(prhs[1]) > 1 && mxIsDouble(prhs[1])) || 
        (nrhs > 4 && !mxIsEmpty(prhs[4]) && mxIsDouble(prhs[4]))){
        check_args(nrhs, prhs, args_real_t, n_real_t, mxDOUBLE_CLASS,
            "double");
        pfdr_d1_ql1b_mex<double, mxDOUBLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }else{
        check_args(nrhs, prhs, args_real_t, n_real_t, mxSINGLE_CLASS,
            "single");
        pfdr_d1_ql1b_mex<float, mxSINGLE_CLASS>(nlhs, plhs, nrhs, prhs);
    }
}

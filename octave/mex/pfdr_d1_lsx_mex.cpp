/*=============================================================================
 * [X, it, Obj, Dif] = pfdr_d1_lsx_mex(loss, Y, edges, edge_weights = 1.0,
 *      loss_weights = [], d1_coor_weights = [], rho = 1.0, cond_min = 1e-2,
 *      dif_rcd = 1e-2, dif_tol = 1e-3, it_max = 1e3, verbose = 1e1)
 * 
 *  Hugo Raguet 2016, 2018
 *===========================================================================*/
#include <cstdint>
#include "mex.h"
#include "../../include/pfdr_d1_lsx.hpp"

using namespace std;

/* vertex_t is an integer type able to represent the number of vertices */
typedef uint32_t vertex_t;
# define VERTEX_CLASS mxUINT32_CLASS
# define VERTEX_ID "uint32"

/* arrays with arguments type */
static const int args_real_t[] = {1, 3, 4, 5};
static const int n_real_t = 4;
static const int args_vertex_t[] = {2};
static const int n_vertex_t = 1;

/* function for checking arguments type */
static void check_args(int nrhs, const mxArray *prhs[], const int* args,
    int n, mxClassID id, const char* id_name)
{
    for (int i = 0; i < n; i++){
        if (nrhs > args[i] && mxGetClassID(prhs[args[i]]) != id
            && mxGetNumberOfElements(prhs[args[i]]) > 1){
            mexErrMsgIdAndTxt("MEX", "PFDR graph d1 loss simplex: argument %d "
                "is of class %s, but class %s is expected.", args[i] + 1,
                mxGetClassName(prhs[args[i]]), id_name);
        }
    }
}

/* resize memory buffer allocated by mxMalloc and create a row vector */
template <typename type_t>
static mxArray* resize_and_create_mxRow(type_t* buffer, size_t size,
    mxClassID id)
{
    mxArray* row = mxCreateNumericMatrix(0, 0, id, mxREAL);
    mxSetM(row, 1);
    mxSetN(row, size);
    buffer = (type_t*) mxRealloc((void*) buffer, sizeof(type_t)*size);
    mxSetData(row, (void*) buffer);
    return row;
}

/* template for handling both single and double precisions */
template<typename real_t>
static void pfdr_d1_lsx_mex(int nlhs, mxArray **plhs, int nrhs,
    const mxArray **prhs)
{
    /**  get inputs  **/

    /* sizes and loss */
    real_t loss = mxGetScalar(prhs[0]);
    size_t D = mxGetM(prhs[1]);
    vertex_t V = mxGetN(prhs[1]);
    const real_t *Y = (real_t*) mxGetData(prhs[1]);
    const real_t *loss_weights = nrhs > 4 && !mxIsEmpty(prhs[4]) ?
        (real_t*) mxGetData(prhs[4]) : nullptr;
    
    /* graph structure */
    check_args(nrhs, prhs, args_vertex_t, n_vertex_t, VERTEX_CLASS, VERTEX_ID);
    size_t E = mxGetNumberOfElements(prhs[2])/2;
    const vertex_t *edges = (vertex_t*) mxGetData(prhs[2]);

    /* penalizations */
    const real_t *edge_weights =
        nrhs > 3 && mxGetNumberOfElements(prhs[3]) > 1 ?
        (real_t*) mxGetData(prhs[3]) : nullptr;
    real_t homo_edge_weight = nrhs > 3 && mxGetNumberOfElements(prhs[3]) == 1 ?
        mxGetScalar(prhs[3]) : 1;
    const real_t *d1_coor_weights = nrhs > 5 && !mxIsEmpty(prhs[5]) ?
        (real_t*) mxGetData(prhs[5]) : nullptr;

    /* algorithmic parameters */
    real_t rho = nrhs > 6 ? mxGetScalar(prhs[6]) : 1.0;
    real_t cond_min = nrhs > 7 ? mxGetScalar(prhs[7]) : 1e-1;
    real_t dif_rcd = nrhs > 8 ? mxGetScalar(prhs[8]) : 1e-2;
    real_t dif_tol = nrhs > 9 ? mxGetScalar(prhs[9]) : 1e-3;
    int it_max = nrhs > 10 ? mxGetScalar(prhs[10]) : 1e3;
    int verbose = nrhs > 11 ? mxGetScalar(prhs[11]) : 1e1;

    /**  create output  **/

    plhs[0] = mxCreateNumericMatrix(D, V, mxGetClassID(prhs[1]), mxREAL);
    real_t *X = (real_t*) mxGetPr(plhs[0]);
    plhs[1] = mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
    int *it = (int*) mxGetData(plhs[1]);

    real_t* Obj = nlhs > 2 ?
        (real_t*) mxMalloc(sizeof(real_t)*(it_max + 1)) : nullptr;
    real_t *Dif = nlhs > 3 ?
        (real_t*) mxMalloc(sizeof(double)*it_max) : nullptr;

    /**  preconditioned forward-Douglas-Rachford  **/

    Pfdr_d1_lsx<real_t, vertex_t> *pfdr = new Pfdr_d1_lsx<real_t, vertex_t>(
        V, E, edges, loss, D, Y, d1_coor_weights);

    pfdr->set_edge_weights(edge_weights, homo_edge_weight);
    pfdr->set_loss(loss_weights);
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
        plhs[2] = resize_and_create_mxRow(Obj, *it + 1, mxGetClassID(prhs[1]));
    }
    if (nlhs > 3){
        plhs[3] = resize_and_create_mxRow(Dif, *it, mxGetClassID(prhs[1]));
    }

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    /* real type is determined by the second parameter Y */
    if (mxIsDouble(prhs[1])){
        check_args(nrhs, prhs, args_real_t, n_real_t, mxDOUBLE_CLASS,
            "double");
        pfdr_d1_lsx_mex<double>(nlhs, plhs, nrhs, prhs);
    }else{
        check_args(nrhs, prhs, args_real_t, n_real_t, mxSINGLE_CLASS,
            "single");
        pfdr_d1_lsx_mex<float>(nlhs, plhs, nrhs, prhs);
    }
}

/*=============================================================================
 * X, it, Obj, Dif = pfdr_d1_ql1b_cpy(
 *          Y, A, edges, edge_weights, Yl1, l1_weights, low_bnd, upp_bnd, rho,
 *          cond_min, dif_rcd, dif_tol, it_max, verbose, AtA_if_square,
 *          real_t_double, compute_Obj, compute_Dif)
 * 
 *  Baudoin Camille 2019
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../include/pfdr_d1_ql1b.hpp" 

using namespace std;

/* vertex_t must be able to represent the number of vertices */ 
typedef uint32_t vertex_t;
# define VERTEX_CLASS NPY_UINT32 
# define VERTEX_ID "uint32"

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES pyREAL_CLASS>
static PyObject* pfdr_d1_ql1b(PyArrayObject* py_Y, PyArrayObject* py_A, 
    PyArrayObject* py_edges, PyArrayObject* py_edge_weights, 
    PyArrayObject* py_Yl1,  PyArrayObject* py_l1_weights, 
    PyArrayObject* py_low_bnd, PyArrayObject* py_upp_bnd, PyArrayObject* py_L,
    real_t rho, real_t cond_min, real_t dif_rcd, real_t dif_tol, int it_max, 
    int verbose, int AtA_if_square, int compute_Obj, int compute_Dif)
{
    /**  get inputs  **/

    /* quadratic functional */
    npy_intp* py_A_dims = PyArray_DIMS(py_A);
    size_t N = py_A_dims[0];
    vertex_t V = PyArray_NDIM(py_A) > 1 ? py_A_dims[1] : 1;

    const real_t *Y = PyArray_SIZE(py_Y) > 0 ?
        (real_t*) PyArray_DATA(py_Y) : nullptr;
    const real_t *A = (N == 1 && V == 1) ?
        nullptr : (real_t*) PyArray_DATA(py_A); 
    real_t * ptr_A = (real_t*) PyArray_DATA(py_A);
    const real_t a = (N == 1 && V == 1) ?
        ptr_A[0] : 1.0; 

    if (V == 1){ /* quadratic functional is only weighted square difference */
        if (N == 1){
            if (PyArray_SIZE(py_Y) > 0){ /* fidelity is square l2 */
                V = PyArray_SIZE(py_Y);
            }else if (PyArray_SIZE (py_Yl1) > 0){
                /* fidelity is only l1 */
                V = PyArray_SIZE(py_Yl1);
            }
        }else{ /* A is given V-by-1, representing a diagonal V-by-V */
            V = N;
        }
        N = DIAG_ATA; /* DIAG_ATA is a macro */
    }else if (V == N && AtA_if_square){
        N = FULL_ATA; 
    }

    /* graph structure */
    size_t E = PyArray_SIZE(py_edges)/2;
    const vertex_t *edges = (vertex_t*) PyArray_DATA(py_edges); 

    /* penalizations */
    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr; 
    real_t * ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1;

    const real_t* Yl1 = (PyArray_SIZE(py_Yl1)>0) ? 
        (real_t*) PyArray_DATA(py_Yl1) : nullptr; 

    const real_t *l1_weights = (PyArray_SIZE(py_l1_weights) > 1) ?
        (real_t*) PyArray_DATA(py_l1_weights) : nullptr;
    real_t * ptr_l1_weights = (real_t*) PyArray_DATA(py_l1_weights);
    real_t homo_l1_weight =  (PyArray_SIZE(py_l1_weights) == 1) ?
        ptr_l1_weights[0] : 0.0;

    const real_t *low_bnd = (PyArray_SIZE(py_low_bnd) > 1) ?
        (real_t*) PyArray_DATA(py_low_bnd) : nullptr; 
    real_t * ptr_low_bnd = (real_t*) PyArray_DATA(py_low_bnd);
    real_t homo_low_bnd = (PyArray_SIZE(py_low_bnd) == 1) ?
        ptr_low_bnd[0] : -INF_REAL;

    const real_t *upp_bnd = (PyArray_SIZE(py_upp_bnd) > 1) ?
        (real_t*) PyArray_DATA(py_upp_bnd) : nullptr; 
    real_t * ptr_upp_bnd = (real_t*) PyArray_DATA(py_upp_bnd);
    real_t homo_upp_bnd = (PyArray_SIZE(py_upp_bnd) == 1) ?
        ptr_upp_bnd[0] : INF_REAL;

    const real_t *L = (PyArray_SIZE(py_L) > 1) ?
        (real_t*) PyArray_DATA(py_L) : nullptr; 
    real_t * ptr_L = (real_t*) PyArray_DATA(py_L);
    real_t l = (PyArray_SIZE(py_L) == 1) ?
        ptr_L[0] : 0.0;

    /**  create output **/

    npy_intp size_py_X[] = {V};
    PyArrayObject* py_X = (PyArrayObject*) PyArray_Zeros(1, size_py_X, 
        PyArray_DescrFromType(pyREAL_CLASS), 1);
    real_t* X = (real_t*) PyArray_DATA(py_X);

    npy_intp size_py_it[] = {1};
    PyArrayObject* py_it = (PyArrayObject*) PyArray_Zeros(1, size_py_it,
        PyArray_DescrFromType(NPY_UINT32), 1);
    int* it = (int*) PyArray_DATA(py_it); 

    real_t* Obj = nullptr;
    PyArrayObject* py_Obj = (PyArrayObject*) Py_None;
    if (compute_Obj){
        npy_intp size_py_Obj[] = {it_max + 1};
        py_Obj = (PyArrayObject*) PyArray_Zeros(1, size_py_Obj,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Obj = (real_t*) PyArray_DATA(py_Obj);
    }

    real_t* Dif = nullptr;
    PyArrayObject* py_Dif = (PyArrayObject*) Py_None;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {it_max};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Dif = (real_t*) PyArray_DATA(py_Dif);
    }

    /**  preconditioned forward-Douglas-Rachford  **/

    Pfdr_d1_ql1b<real_t, vertex_t> *pfdr =
        new Pfdr_d1_ql1b<real_t, vertex_t>(V, E, edges);

    pfdr->set_edge_weights(edge_weights, homo_edge_weight);
    pfdr->set_quadratic(Y, N, A, a);
    pfdr->set_l1(l1_weights, homo_l1_weight, Yl1);
    pfdr->set_bounds(low_bnd, homo_low_bnd, upp_bnd, homo_upp_bnd);
    if (PyArray_SIZE(py_L) != 0) {pfdr->set_lipschitz_param(L, l);}
    pfdr->set_conditioning_param(cond_min, dif_rcd);
    pfdr->set_relaxation(rho);
    pfdr->set_algo_param(dif_tol, it_max, verbose);
    pfdr->set_monitoring_arrays(Obj, Dif);
    pfdr->set_iterate(X);
    pfdr->initialize_iterate();

    *it = pfdr->precond_proximal_splitting();
    
    pfdr->set_iterate(nullptr); // prevent X to be free()'d
    
    delete pfdr;
    
    return Py_BuildValue("OOOO", py_X, py_it, py_Obj, py_Dif); 
}

/* My python wrapper */
static PyObject* pfdr_d1_ql1b_cpy(PyObject * self, PyObject * args)
{ 
    /* My INPUT */ 
    PyArrayObject *py_Y, *py_A, *py_edges, *py_edge_weights, *py_Yl1, 
        *py_l1_weights, *py_low_bnd, *py_upp_bnd, *py_L; 
    double rho, cond_min, dif_rcd, dif_tol;
    int it_max, verbose, AtA_if_square, real_t_double, compute_Obj, 
        compute_Dif; 
    
    /* parse the input, from Python Object to C PyArray, double, or int type */
#if PY_MAJOR_VERSION >= 3
    if(!PyArg_ParseTuple(args, "OOOOOOOOOddddiipppp", &py_Y, &py_A,
#else // python 2 does not accept the 'p' format specifier
    if(!PyArg_ParseTuple(args, "OOOOOOOOOddddiiiiii", &py_Y, &py_A,
#endif
        &py_edges, &py_edge_weights, &py_Yl1, &py_l1_weights, &py_low_bnd, 
        &py_upp_bnd, &py_L, &rho, &cond_min, &dif_rcd, &dif_tol, &it_max,
        &verbose, &AtA_if_square, &real_t_double, &compute_Obj, 
        &compute_Dif)) {
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        PyObject* PyReturn = pfdr_d1_ql1b<double, NPY_FLOAT64>(py_Y, py_A, 
            py_edges, py_edge_weights, py_Yl1, py_l1_weights, py_low_bnd, 
            py_upp_bnd, py_L, rho, cond_min, dif_rcd, dif_tol, it_max,
            verbose, AtA_if_square, compute_Obj, compute_Dif);
        return PyReturn;
    }else{ /* real_t type is float */
        PyObject* PyReturn = pfdr_d1_ql1b<float, NPY_FLOAT32>(py_Y, py_A,
            py_edges, py_edge_weights, py_Yl1, py_l1_weights, py_low_bnd, 
            py_upp_bnd, py_L, rho, cond_min, dif_rcd, dif_tol, it_max, verbose,
            AtA_if_square, compute_Obj, compute_Dif);
        return PyReturn;
    }
}

static PyMethodDef pfdr_d1_ql1b_methods[] = {
    {"pfdr_d1_ql1b_cpy", pfdr_d1_ql1b_cpy, METH_VARARGS,
        "wrapper for PFDR quadratic d1 l1 bounds"},
    {NULL, NULL, 0, NULL}
};

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef pfdr_d1_ql1b_module = {
    PyModuleDef_HEAD_INIT,
    "pfdr_d1_ql1b_cpy", /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    pfdr_d1_ql1b_methods,
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_pfdr_d1_ql1b_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&pfdr_d1_ql1b_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initpfdr_d1_ql1b_cpy(void)
{
    (void) Py_InitModule("pfdr_d1_ql1b_cpy", pfdr_d1_ql1b_methods);
    import_array() /* IMPORTANT: this must be called to use numpy array */
}

#endif

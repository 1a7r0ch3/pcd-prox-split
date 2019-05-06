/*=============================================================================
 * Comp, rX, it, Obj,  Dif = pfdr_d1_lsx_cpy(
 *          loss, Y, edges, edge_weights, loss_weights, d1_coor_weights, rho, 
 *          cond_min, dif_rcd, dif_tol, it_max, verbose, real_t_double,
 *          compute_Obj, compute_Dif)
 * 
 *  Baudoin Camille 2019
 *===========================================================================*/
#include <cstdint>
#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../include/pfdr_d1_lsx.hpp"

using namespace std;

/* vertex_t must be able to represent the number of vertices and of (undirected)
 * edges in the main graph;
 * comp_t must be able to represent the number of constant connected components
 * in the reduced graph, as well as the dimension D */
typedef uint32_t vertex_t;
# define VERTEX_CLASS NPY_UINT32
# define VERTEX_ID "uint32"

/* template for handling both single and double precisions */
template<typename real_t, NPY_TYPES pyREAL_CLASS>
static PyObject* pfdr_d1_lsx(real_t loss, PyArrayObject* py_Y,
    PyArrayObject* py_edges, PyArrayObject* py_edge_weights, 
    PyArrayObject* py_loss_weights, PyArrayObject* py_d1_coor_weights,
    real_t rho, real_t cond_min, real_t dif_rcd,
    real_t dif_tol, int it_max, int verbose, int compute_Obj,
    int compute_Dif)
{
    /**  get inputs  **/
    /* sizes and loss */
    npy_intp * py_Y_size = PyArray_DIMS(py_Y);
    size_t D = py_Y_size[0];
    vertex_t V = py_Y_size[1]; 

    const real_t *Y = (real_t*) PyArray_DATA(py_Y);
    const real_t *loss_weights = (PyArray_SIZE(py_loss_weights) > 0) ?
        (real_t*) PyArray_DATA(py_loss_weights) : nullptr;

    /* graph structure */
    vertex_t E = PyArray_SIZE(py_edges) / 2;
    const vertex_t *edges = (vertex_t*) PyArray_DATA(py_edges);

    /* penalizations */
    const real_t *edge_weights = (PyArray_SIZE(py_edge_weights) > 1) ?
        (real_t*) PyArray_DATA(py_edge_weights) : nullptr;
    real_t* ptr_edge_weights = (real_t*) PyArray_DATA(py_edge_weights);
    real_t homo_edge_weight = (PyArray_SIZE(py_edge_weights) == 1) ?
        ptr_edge_weights[0] : 1;
    const real_t* d1_coor_weights = (PyArray_SIZE(py_d1_coor_weights) > 0) ?
        (real_t*) PyArray_DATA(py_d1_coor_weights) : nullptr;

    /**  create output  **/

    npy_intp size_py_X[] = {(npy_intp) D, V};
    PyArrayObject* py_X = (PyArrayObject*) PyArray_Zeros(2, size_py_X, 
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
    PyArrayObject *py_Dif = (PyArrayObject*) Py_None;
    if (compute_Dif){
        npy_intp size_py_Dif[] = {it_max};
        py_Dif = (PyArrayObject*) PyArray_Zeros(1, size_py_Dif,
            PyArray_DescrFromType(pyREAL_CLASS), 1);
        Dif = (real_t*) PyArray_DATA(py_Dif);
    }    

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

    return Py_BuildValue("OOOO", py_X, py_it, py_Obj, py_Dif);
}

/* My python wrapper */
static PyObject* pfdr_d1_lsx_cpy(PyObject* self, PyObject* args)
{ 
    /* My INPUT */
    PyArrayObject *py_Y, *py_edges, *py_edge_weights, *py_loss_weights, 
        *py_d1_coor_weights;
    double loss, rho, cond_min, dif_rcd, dif_tol;  
    int it_max, verbose, real_t_double, compute_Obj, compute_Dif;

    /* parse the input, from Python Object to C PyArray, double, or int type */
#if PY_MAJOR_VERSION >= 3
    if(!PyArg_ParseTuple(args, "dOOOOOddddiippp", &loss, &py_Y,
#else // python 2 does not accept the 'p' format specifier
    if(!PyArg_ParseTuple(args, "dOOOOOddddiiiii", &loss, &py_Y,
#endif
        &py_edges, &py_edge_weights, &py_loss_weights, &py_d1_coor_weights,  
        &rho, &cond_min, &dif_rcd, &dif_tol, &it_max, &verbose, &real_t_double,
        &compute_Obj, &compute_Dif)){
        return NULL;
    }

    if (real_t_double){ /* real_t type is double */
        PyObject* PyReturn = pfdr_d1_lsx<double, NPY_FLOAT64>(loss, py_Y,
            py_edges, py_edge_weights, py_loss_weights, py_d1_coor_weights, 
            rho, cond_min, dif_rcd, dif_tol, it_max, verbose, compute_Obj,
            compute_Dif);
        return PyReturn;
    }else{ /* real_t type is float */
        PyObject* PyReturn = pfdr_d1_lsx<float, NPY_FLOAT32>(loss, py_Y,
            py_edges, py_edge_weights, py_loss_weights, py_d1_coor_weights, 
            rho, cond_min, dif_rcd, dif_tol, it_max, verbose, compute_Obj, 
            compute_Dif);
        return PyReturn;
    }
}

static PyMethodDef pfdr_d1_lsx_methods[] = {
    {"pfdr_d1_lsx_cpy", pfdr_d1_lsx_cpy, METH_VARARGS,
        "wrapper for pfdr loss d1 simplex"},
    {NULL, NULL, 0, NULL}
}; 

/* module initialization */
#if PY_MAJOR_VERSION >= 3
/* Python version 3 */
static struct PyModuleDef pfdr_d1_lsx_module = {
    PyModuleDef_HEAD_INIT,
    "pfdr_d1_lsx_cpy", /* name of module */
    NULL, /* module documentation, may be null */
    -1,   /* size of per-interpreter state of the module,
             or -1 if the module keeps state in global variables. */
    pfdr_d1_lsx_methods, /* actual methods in the module */
    NULL, /* multi-phase initialization, may be null */
    NULL, /* traversal function, may be null */
    NULL, /* clearing function, may be null */
    NULL  /* freeing function, may be null */
};

PyMODINIT_FUNC
PyInit_pfdr_d1_lsx_cpy(void)
{
    import_array() /* IMPORTANT: this must be called to use numpy array */
    return PyModule_Create(&pfdr_d1_lsx_module);
}

#else

/* module initialization */
/* Python version 2 */
PyMODINIT_FUNC
initpfdr_d1_lsx_cpy(void)
{
    (void) Py_InitModule("pfdr_d1_lsx_cpy", pfdr_d1_lsx_methods);
    import_array() /* IMPORTANT: this must be called to use numpy array */
}

#endif

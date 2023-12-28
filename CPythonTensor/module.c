#include "module.h"
#include <Python.h>
#include <numpy/arrayobject.h>

typedef struct
{
    PyObject_HEAD
    void* t;
} PyTensorObject;

static int Tensor_init(PyTensorObject* self, PyObject* args)
{
    PyObject *input_array;
    PyArg_ParseTuple(args, "O", &input_array);

    PyArrayObject* np_array = PyArray_FROM_OTF(input_array, NPY_FLOAT, NPY_ARRAY_IN_ARRAY);
    if (np_array == NULL) {
        return NULL; // Error handling for NumPy array conversion
    }

    int ndims = PyArray_NDIM(np_array);
    npy_intp* dims = PyArray_DIMS(np_array);
    unsigned int* c_dims = calloc(ndims, sizeof(unsigned int));
    for (size_t i = 0; i < ndims; i++)
    {
        c_dims[i] = dims[i];
    }

    if (self != NULL) {
        self->t = call_tensor(ndims, c_dims, PyArray_DATA(np_array));
    }
    free(c_dims);
    return self;
}

static PyObject* Tensor_new(PyTypeObject* type, PyObject* args)
{
    PyTensorObject* self;
    self = type->tp_alloc(type, 0);
    if (self != NULL) {
        self->t = 0;
    }
    return self;
}

static void Tensor_dealloc(PyTensorObject* self)
{
    delete_tensor(self->t);
    Py_TYPE(self)->tp_free(self);
}

static PyObject*
Tensor_ToString(PyTensorObject* self)
{
    return PyUnicode_FromString(to_string(self->t));
}

static PyMethodDef Tensor_methods[] = {
    {NULL}  /* Sentinel */
};

static PyTypeObject TensorType =
{
    .ob_base = PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "tensor.Tensor",
    .tp_doc = PyDoc_STR("Custom objects"),
    .tp_basicsize = sizeof(PyTensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_init = Tensor_init,
    .tp_new = Tensor_new,
    .tp_dealloc = Tensor_dealloc,
    .tp_methods = Tensor_methods,
    .tp_str = Tensor_ToString,
};

static PyModuleDef tensor_module =
{
    PyModuleDef_HEAD_INIT,
    "tensor",
    NULL,
    -1
};

PyMODINIT_FUNC
PyInit_tensor()
{
    import_array();
    PyObject* m;
    if (PyType_Ready(&TensorType) < 0)
        return NULL;

    m = PyModule_Create(&tensor_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", &TensorType) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}

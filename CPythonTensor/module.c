#include "module.h"
#include <Python.h>

typedef struct
{
    PyObject_HEAD
    void* t;
} PyTensorObject;

static PyObject* Tensor_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    static char* kwlist[] = { "dtype", "shape", "data", NULL};
    PyObject *dtype, *shape;
    Py_ssize_t data_ptr;
    PyArg_ParseTupleAndKeywords(args, kwds, "|OOn", kwlist, &dtype, &shape, &data_ptr);
    unsigned int* ca = calloc(PyTuple_GET_SIZE(shape), sizeof(unsigned int));
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(shape); i++)
        ca[i] = _PyLong_AsInt(PyTuple_GET_ITEM(shape, i));
    PyTensorObject* self;
    self = type->tp_alloc(type, 0);
    if (self != NULL) {
        self->t = call_tensor(PyTuple_GET_SIZE(shape), ca, data_ptr);
    }
    free(ca);
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

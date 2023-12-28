#include "module.h"
#include <sstream>
#include <tensor.hh>

using namespace tensor_array::value;

void* call_tensor(unsigned int nd, unsigned int* dimensions, const void* data)
{
    return new Tensor(TensorBase(typeid(float), std::initializer_list(dimensions, dimensions + nd), data, tensor_array::devices::DEVICE_CPU_0));
}

void delete_tensor(void* t)
{
    delete t;
}

const char* to_string(void* t)
{
    Tensor* t1 = static_cast<Tensor*>(t);
    std::ostringstream steam;
    steam << *t1;
    return steam.str().c_str();
}

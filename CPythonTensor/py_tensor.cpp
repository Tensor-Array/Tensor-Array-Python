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

void* add_tensor(const void* a, const void* b)
{
    const Tensor* t_a = static_cast<const Tensor*>(a);
    const Tensor* t_b = static_cast<const Tensor*>(b);
    return new Tensor(add(*t_a, *t_b));
}

const char* to_string(void* t)
{
    Tensor* t1 = static_cast<Tensor*>(t);
    std::ostringstream stream;
    stream << *t1;
    return stream.str().c_str();
}

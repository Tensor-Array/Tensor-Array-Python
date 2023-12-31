#include <tensor-array/core/tensor.hh>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

using namespace tensor_array::value;

template <typename T>
TensorBase convert_numpy_to_tensor_base(pybind11::array_t<T> py_buf)
{
	pybind11::buffer_info info = py_buf.request();
	std::vector<unsigned int> shape_vec(info.ndim);
	std::transform
	(
		info.shape.cbegin(),
		info.shape.cend(),
		shape_vec.begin(),
		[](pybind11::size_t dim)
		{
			return static_cast<unsigned int>(dim);
		}
	);
	return TensorBase(typeid(T), shape_vec, info.ptr);
}

std::string tensor_to_string(const Tensor t)
{
	std::ostringstream osstream;
	osstream << t;
	return osstream.str();
}

PYBIND11_MODULE(tensor, m)
{
	pybind11::class_<Tensor>(m, "Tensor")
		.def(pybind11::init())
		.def(pybind11::init(&convert_numpy_to_tensor_base<float>))
		.def(pybind11::self + pybind11::self)
		.def(pybind11::self - pybind11::self)
		.def(pybind11::self * pybind11::self)
		.def(pybind11::self / pybind11::self)
		.def(pybind11::self += pybind11::self)
		.def(pybind11::self -= pybind11::self)
		.def(pybind11::self *= pybind11::self)
		.def(pybind11::self /= pybind11::self)
		.def(pybind11::self == pybind11::self)
		.def(pybind11::self != pybind11::self)
		.def(pybind11::self >= pybind11::self)
		.def(pybind11::self <= pybind11::self)
		.def(pybind11::self > pybind11::self)
		.def(pybind11::self < pybind11::self)
		.def(+pybind11::self)
		.def(-pybind11::self)
		.def("__matmul__", &matmul)
		.def("__repr__", &tensor_to_string);
}
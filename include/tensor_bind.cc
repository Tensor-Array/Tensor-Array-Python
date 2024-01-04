#include <tensor-array/core/tensor.hh>
#include <tensor-array/core/data_type_wrapper.hh>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>

using namespace tensor_array::value;
using namespace tensor_array::datatype;

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
	warp_type(warp_type(typeid(T)));
	return TensorBase(typeid(T), shape_vec, info.ptr);
}

pybind11::dtype get_py_type(const std::type_info& info)
{
	if (info == typeid(bool))
		return pybind11::dtype::of<bool>();
	if (info == typeid(float))
		return pybind11::dtype::of<float>();
	throw std::exception();
}

pybind11::array convert_tensor_to_numpy(const Tensor& tensor)
{
	const TensorBase& base_tensor = tensor.get_buffer().change_device({tensor_array::devices::CPU, 0});
	std::vector<pybind11::size_t> shape_vec(base_tensor.shape().size());
	std::transform
	(
		base_tensor.shape().begin(),
		base_tensor.shape().end(),
		shape_vec.begin(),
		[](unsigned int dim)
		{
			return static_cast<pybind11::size_t>(dim);
		}
	);
	auto ty0 = pybind11::detail::get_type_info(base_tensor.type());
	pybind11::dtype ty1 = get_py_type(base_tensor.type());
	return pybind11::array(ty1, shape_vec, base_tensor.data());
}

Tensor python_tuple_slice(const Tensor& t, pybind11::tuple tuple_slice)
{
	std::vector<Tensor::Slice> t_slices;
	for (size_t i = 0; i < tuple_slice.size(); i++)
	{
		ssize_t start, stop, step;
		ssize_t length;
		pybind11::slice py_slice = tuple_slice[i].cast<pybind11::slice>();
		if (!py_slice.compute(t.get_buffer().shape().begin()[i], &start, &stop, &step, &length))
			throw std::runtime_error("Invalid slice");
		t_slices.insert
		(
			t_slices.begin() + i,
			Tensor::Slice
			{
				static_cast<int>(start),
				static_cast<int>(stop),
				static_cast<int>(step)
			}
		);
	}
	return t[tensor_array::wrapper::initializer_wrapper(t_slices.begin().operator->(), t_slices.end().operator->())];
}

Tensor python_slice(const Tensor& t, pybind11::slice py_slice)
{
	std::vector<Tensor::Slice> t_slices;
	ssize_t start, stop, step;
	ssize_t length;
	if (!py_slice.compute(t.get_buffer().shape().begin()[0], &start, &stop, &step, &length))
	throw std::runtime_error("Invalid slice");
	return t
	[
		{
			Tensor::Slice
			{
				static_cast<int>(start),
				static_cast<int>(stop),
				static_cast<int>(step)
			}
		}
	];
}

Tensor python_index(const Tensor& t, unsigned int i)
{
	return t[i];
}

std::size_t python_len(const Tensor& t)
{
	std::initializer_list<unsigned int> shape_list = t.get_buffer().shape();
	return shape_list.size() != 0 ? shape_list.begin()[0]: 1U;
}

pybind11::str tensor_to_string(const Tensor& t)
{
	return pybind11::repr(convert_tensor_to_numpy(t));
}

Tensor tensor_cast_1(const Tensor& t, DataType dtype)
{
	return t.tensor_cast(warp_type(dtype));
}

PYBIND11_MODULE(tensor2, m)
{
	pybind11::enum_<DataType>(m, "DataType")
		.value("BOOL", BOOL_DTYPE)
		.value("S_INT_8", S_INT_8)
		.value("S_INT_16", S_INT_16)
		.value("S_INT_32", S_INT_32)
		.value("S_INT_64", S_INT_64)
		.value("FLOAT", FLOAT_DTYPE)
		.value("DOUBLE", DOUBLE_DTYPE)
		.value("HALF", HALF_DTYPE)
		.value("BFLOAT16", BF16_DTYPE)
		.value("U_INT_8", U_INT_8)
		.value("U_INT_16", U_INT_16)
		.value("U_INT_32", U_INT_32)
		.value("U_INT_64", U_INT_64)
		.export_values();

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
		.def(hash(pybind11::self))
		.def("transpose", &Tensor::transpose)
		.def("calc_grad", &Tensor::calc_grad)
		.def("get_grad", &Tensor::get_grad)
		.def("sin", &Tensor::sin)
		.def("sin", &Tensor::sin)
		.def("cos", &Tensor::cos)
		.def("tan", &Tensor::tan)
		.def("sinh", &Tensor::sinh)
		.def("cosh", &Tensor::cosh)
		.def("tanh", &Tensor::tanh)
		.def("log", &Tensor::log)
		.def("clone", &Tensor::clone)
		.def("cast", &tensor_cast_1)
		.def("add", &add)
		.def("multiply", &multiply)
		.def("divide", &divide)
		.def("matmul", &matmul)
		.def("condition", &condition)
		.def("numpy", &convert_tensor_to_numpy)
		.def("__getitem__", &python_index)
		.def("__getitem__", &python_slice)
		.def("__getitem__", &python_tuple_slice)
		.def("__len__", &python_len)
		.def("__matmul__", &matmul)
		.def("__rmatmul__", &matmul)
		.def("__repr__", &tensor_to_string);
}
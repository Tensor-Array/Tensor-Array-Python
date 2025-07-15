#include <tensor-array/core/tensor.hh>
#include <tensor-array/core/data_type_wrapper.hh>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>

using namespace tensor_array::value;
using namespace tensor_array::datatype;
using namespace tensor_array::wrapper;

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
	return TensorBase(warp_type(warp_type(typeid(T))), shape_vec, info.ptr);
}

pybind11::dtype get_py_type(const std::type_info& info)
{
	if (info == typeid(std::int8_t))
		return pybind11::dtype::of<std::int8_t>();
	if (info == typeid(std::int16_t))
		return pybind11::dtype::of<std::int16_t>();
	if (info == typeid(std::int32_t))
		return pybind11::dtype::of<std::int32_t>();
	if (info == typeid(std::int64_t))
		return pybind11::dtype::of<std::int64_t>();
	if (info == typeid(std::uint8_t))
		return pybind11::dtype::of<std::uint8_t>();
	if (info == typeid(std::uint16_t))
		return pybind11::dtype::of<std::uint16_t>();
	if (info == typeid(std::uint32_t))
		return pybind11::dtype::of<std::uint32_t>();
	if (info == typeid(std::uint64_t))
		return pybind11::dtype::of<std::uint64_t>();
	if (info == typeid(bool))
		return pybind11::dtype::of<bool>();
	if (info == typeid(float))
		return pybind11::dtype::of<float>();
	if (info == typeid(double))
		return pybind11::dtype::of<double>();
	throw std::runtime_error("no dtype");
}

pybind11::array convert_tensor_to_numpy(const Tensor& self)
{
	const TensorBase& base_tensor = self.get_buffer().change_device({tensor_array::devices::CPU, 0});
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

Tensor python_tuple_slice(const Tensor& self, pybind11::tuple tuple_slice)
{
	std::vector<Tensor::Slice> t_slices;
	for (size_t i = 0; i < tuple_slice.size(); i++)
	{
		ssize_t start, stop, step;
		ssize_t length;
		pybind11::slice py_slice = tuple_slice[i].cast<pybind11::slice>();
		if (!py_slice.compute(self.get_buffer().shape().begin()[i], &start, &stop, &step, &length))
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
	return self[initializer_wrapper<Tensor::Slice>(t_slices.begin().operator->(), t_slices.end().operator->())];
}

Tensor python_slice(const Tensor& self, pybind11::slice py_slice)
{
	std::vector<Tensor::Slice> t_slices;
	ssize_t start, stop, step;
	ssize_t length;
	if (!py_slice.compute(self.get_buffer().shape().begin()[0], &start, &stop, &step, &length))
	throw std::runtime_error("Invalid slice");
	return self
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

Tensor python_index(const Tensor& self, unsigned int i)
{
	return self[i];
}

std::size_t python_len(const Tensor& self)
{
	std::initializer_list<unsigned int> shape_list = self.get_buffer().shape();
	return shape_list.size() != 0 ? shape_list.begin()[0]: 1U;
}

pybind11::str tensor_to_string(const Tensor& self)
{
	return pybind11::repr(convert_tensor_to_numpy(self));
}

Tensor tensor_cast_1(const Tensor& self, DataType dtype)
{
	return self.tensor_cast(warp_type(dtype));
}

pybind11::tuple tensor_shape(const Tensor& self)
{
	return pybind11::cast(std::vector<unsigned int>(self.get_buffer().shape()));
}

DataType tensor_type(const Tensor& self)
{
	return warp_type(self.get_buffer().type());
}

Tensor tensor_copying(const Tensor& self)
{
	return Tensor(self);
}

Tensor py_zeros(pybind11::tuple shape_tuple, DataType dtype)
{
	std::vector<unsigned int> shape_vec;
	for (auto& it: shape_tuple)
		shape_vec.push_back(it.cast<unsigned int>());
	return TensorBase(warp_type(dtype), shape_vec);
}

Tensor py_rand(pybind11::tuple shape_tuple, unsigned int seed = std::rand())
{
	std::vector<unsigned int> shape_vec;
	for (auto& it: shape_tuple)
		shape_vec.push_back(it.cast<unsigned int>());
	return tensor_rand(shape_vec, seed);
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
	
	m.def
	(
		"zeros",
		&py_zeros,
		pybind11::arg("shape"),
		pybind11::arg("dtype") = S_INT_32
	);

	m.def(
		"rand",
		&py_rand,
		pybind11::arg("shape"),
		pybind11::arg("seed") = std::rand()
	);

	m.def(
		"add",
		&tensor_array::value::add,
		pybind11::arg("value_1"),
		pybind11::arg("value_2")
	);

	m.def(
		"multiply",
		&tensor_array::value::multiply,
		pybind11::arg("value_1"),
		pybind11::arg("value_2")
	);

	m.def(
		"divide",
		&tensor_array::value::divide,
		pybind11::arg("value_1"),
		pybind11::arg("value_2")
	);

	m.def(
		"power",
		&tensor_array::value::power,
		pybind11::arg("value_1"),
		pybind11::arg("value_2")
	);
	
	m.def(
		"matmul",
		&tensor_array::value::matmul,
		pybind11::arg("value_1"),
		pybind11::arg("value_2")
	);

	m.def(
		"condition",
		&tensor_array::value::condition,
		pybind11::arg("condition_value"),
		pybind11::arg("value_if_true"),
		pybind11::arg("value_if_false")
	);

	pybind11::class_<Tensor>(m, "Tensor")
		.def(pybind11::init())
		.def(pybind11::init(&tensor_copying))
		.def(pybind11::init(&convert_numpy_to_tensor_base<int>))
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
		.def("cos", &Tensor::cos)
		.def("tan", &Tensor::tan)
		.def("sinh", &Tensor::sinh)
		.def("cosh", &Tensor::cosh)
		.def("tanh", &Tensor::tanh)
		.def("log", &Tensor::log)
		.def("clone", &Tensor::clone)
		.def("cast", &tensor_cast_1)
		.def("numpy", &convert_tensor_to_numpy)
		.def("shape", &tensor_shape)
		.def("dtype", &tensor_type)
		.def("__getitem__", &python_index)
		.def("__getitem__", &python_slice)
		.def("__getitem__", &python_tuple_slice)
		.def("__len__", &python_len)
		.def("__matmul__", &matmul)
		.def("__repr__", &tensor_to_string)
		.def("__copy__", &tensor_copying);
}
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

pybind11::array convert_tensor_to_numpy(const Tensor& tensor)
{
	const TensorBase& base_tensor = tensor.get_buffer();
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
	pybind11::array arr = pybind11::array();
	return arr;
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
	
	#ifdef __GNUC__
	struct
	{
		const Tensor::Slice* it;
		std::size_t sz;
	} test;
	test.it = t_slices.data();
	test.sz = t_slices.size();
	std::initializer_list<Tensor::Slice>& t_slice_list = reinterpret_cast<std::initializer_list<Tensor::Slice>&>(test);
	#endif
	return t[t_slice_list];
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

std::string tensor_to_string(const Tensor& t)
{
	std::ostringstream osstream;
	osstream << t;
	return osstream.str();
}

PYBIND11_MODULE(tensor2, m)
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
		.def(hash(pybind11::self))
		.def("transpose", &Tensor::transpose)
		.def("calc_grad", &Tensor::calc_grad)
		.def("add", &add)
		.def("multiply", &multiply)
		.def("divide", &divide)
		.def("matmul", &matmul)
		.def("condition", &condition)
		.def("__getitem__", &python_index)
		.def("__getitem__", &python_slice)
		.def("__getitem__", &python_tuple_slice)
		.def("__len__", &python_len)
		.def("__matmul__", &matmul)
		.def("__repr__", &tensor_to_string);
}
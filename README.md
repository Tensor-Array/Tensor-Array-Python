# Tensor Array Python
[![pypi](https://img.shields.io/pypi/v/TensorArray)](https://pypi.org/project/TensorArray/)
[![status](https://img.shields.io/pypi/status/TensorArray)](https://pypi.org/project/TensorArray/)
[![python](https://img.shields.io/pypi/pyversions/TensorArray)](https://pypi.org/project/TensorArray/)
[![download per month](https://img.shields.io/pypi/dm/TensorArray)](https://pypi.org/project/TensorArray/)
[![license](https://img.shields.io/pypi/l/TensorArray)](#)

This machine learning library using [Tensor-Array](https://github.com/Tensor-Array/Tensor-Array) library

This project is still in alpha version, we are trying to make this look like the main framework but it is easier to code.

## How to install Tensor-Array python version.

Before install this library please install [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) first.

It can not work without [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit).

If you did not install [Python](https://www.python.org/) then install [Python 3](https://www.python.org/):

```shell
apt-get update
apt-get install python3
```

After that go to command and install:

```shell
pip install TensorArray
```

## Testing with the [Tensor](https://github.com/Tensor-Array/Tensor-Array/tab=readme-ov-file#the-tensor-class) object.

The `Tensor` class is a storage that store value and calculate the tensor.

The `Tensor.calc_grad()` method can do automatic differentiation.

The `Tensor.get_grad()` method can get the gradient after call `Tensor.calc_grad()`.

```python
import tensor_array.core as ta
import numpy as np

def test_add():
    example_tensor_array = ta.Tensor(np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
        ], dtype=np.int32))
    example_tensor_array_scalar = ta.Tensor(100)
    example_tensor_sum = example_tensor_array + example_tensor_array_scalar
    print(example_tensor_sum)
    example_tensor_sum.calc_grad()
    print(example_tensor_array.get_grad())
    print(example_tensor_array_scalar.get_grad())

test_add()

```

from tensor_array.core import Tensor
from tensor_array.core import zeros
from tensor_array.core import condition

def relu(input: Tensor) -> Tensor:
    tensor_zeros = zeros(shape = input.shape(), dtype = input.dtype())
    return condition(input > tensor_zeros, input, tensor_zeros)

def sigmoid(input: Tensor) -> Tensor:
    return input.sigmoid()

def softmax(input: Tensor, dim: int = 0) -> Tensor:
    return input.softmax(dim)

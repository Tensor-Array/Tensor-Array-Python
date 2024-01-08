from tensor_array.core import Tensor
from tensor_array.core import zeros

def relu(input):
    tensor_zeros = zeros(shape = input.shape(), dtype = input.dtype())
    return (input > tensor_zeros).condition(input, tensor_zeros)

def sigmoid(input):
    return input.sigmoid()

def softmax(input, dim = 0):
    return input

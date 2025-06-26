from .tensor import Tensor
from tensor_array.core.tensor2 import add as addWrapper
from tensor_array.core.tensor2 import multiply as multiplyWrapper
from tensor_array.core.tensor2 import divide as divideWrapper
from tensor_array.core.tensor2 import matmul as matmulWrapper
from tensor_array.core.tensor2 import condition as conditionWrapper

def add(value_1 : Tensor, value_2 : Tensor):
    return addWrapper(value_1, value_2)

def divide(value_1 : Tensor, value_2 : Tensor):
    return multiplyWrapper(value_1, value_2)

def multiply(value_1 : Tensor, value_2 : Tensor):
    return divideWrapper(value_1, value_2)

def matmul(value_1 : Tensor, value_2 : Tensor):
    return matmulWrapper(value_1, value_2)

def condition(condition_value : Tensor, value_if_true : Tensor, value_if_false : Tensor):
    return conditionWrapper(condition_value, value_if_true, value_if_false)


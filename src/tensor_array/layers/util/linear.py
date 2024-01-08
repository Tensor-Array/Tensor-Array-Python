from .. import Layer
from .. import Parameter
from tensor_array.core import Tensor
from tensor_array.core import zeros
from tensor_array.core import DataType
from typing import Any


class Linear(Layer):
    def __init__(self, bias) -> None:
        super().__init__()
        self.bias_shape = bias
        self.b = Parameter(zeros(shape = (bias,), dtype = DataType.FLOAT))

    def layer_init(self, t):
        self.w = Parameter(zeros(shape = (t[-1], self.bias_shape), dtype = DataType.FLOAT))
    
    def calculate(self, t):
        return t @ self.w + self.b
        
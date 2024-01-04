from tensor_array.layers import Layer
from tensor_array.layers import Parameter
from tensor_array.core import Tensor
from typing import Any


class Linear(Layer):
    def __init__(self, bias) -> None:
        super(self)
        self.bias_shape = bias
        self.b = Parameter(Tensor(shape = (bias)))

    def init_value(self, t):
        self.w = Parameter(Tensor(shape = (t.shape(-1), self.bias_shape)))
    
    def calculate(self, t):
        return self.w @ t + self.b
        
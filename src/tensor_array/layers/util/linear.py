from .. import Layer
from .. import Parameter
from tensor_array.core import Tensor
from tensor_array.core import zeros
from typing import Any


class Linear(Layer):
    def __init__(self, bias) -> None:
        super(Linear, self).__init__()
        self.bias_shape = bias
        self.b = Parameter(zeros(shape = (bias,)))

    def init_value(self, t):
        self.w = Parameter(zeros(shape = (t.shape()[-1], self.bias_shape)))
    
    def calculate(self, t):
        print("t", t)
        print("w", self.w)
        print("b", self.b)
        return t @ self.w + self.b
        
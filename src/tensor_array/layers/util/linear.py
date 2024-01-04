from tensor_array.layers import Layer
from tensor_array.core import Tensor
from typing import Any


class Linear(Layer):
    def __init__(self) -> None:
        self.w = t.Tensor(0)
        self.b = t.Tensor(0)

    def __call__(self, input) -> Any:
        return input @ self.w + self.b
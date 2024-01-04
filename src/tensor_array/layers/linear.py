from TensorArray.core import tensor2 as t
from typing import Any


class Linear:
    def __init__(self) -> None:
        self.w = t.Tensor(0)
        self.b = t.Tensor(0)

    def __call__(self, input) -> Any:
        return input @ self.w + self.b
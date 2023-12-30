from typing import Self
import numpy as np
import tensor as t

class Tensor:
    def __init__(self, arr, *args):
        self.temp_tensor = t.TensorC(arr)
        
    def __add__(self, other) -> Self:
        result = Tensor(0);
        result.temp_tensor = self.temp_tensor + other.temp_tensor
        return result;

    def __matmul__(self, other) -> Self:
        result = Tensor(0);
        result.temp_tensor = self.temp_tensor @ other.temp_tensor
        return result;

    def __str__(self) -> str:
        return self.temp_tensor.__str__()



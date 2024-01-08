from .. import Layer
from .. import Parameter
from tensor_array.core import Tensor
from tensor_array.core import zeros
from tensor_array.core import DataType
from typing import Any, List, OrderedDict


class Sequential(Layer):
    def __init__(self, _layers: OrderedDict[str, Layer]) -> None:
        self._layers = _layers
    
    def calculate(self, t):
        tensorloop = t
        for _, content in self._layers:
            tensorloop = content(tensorloop)
        return tensorloop
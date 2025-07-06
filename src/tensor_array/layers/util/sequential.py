from .. import Layer
from .. import Parameter
from tensor_array.core import Tensor
from tensor_array.core import zeros
from tensor_array.core import DataTypes
from typing import Any, List, OrderedDict


class Sequential(Layer):
    def __init__(self, _layers: OrderedDict[str, Layer]) -> None:
        """
        Initializes a Sequential layer with a list of layers.
        Args:
            _layers (OrderedDict[str, Layer]): An ordered dictionary of layers to be applied sequentially.
        """
        self._layers = _layers
    
    def calculate(self, t):
        """
        Applies each layer in the sequential model to the input tensor in order.
        Args:
            t (Tensor): The input tensor to be processed through the layers.
        Returns:
            Tensor: The output tensor after passing through all layers.
        """
        tensorloop = t
        for _, content in self._layers:
            tensorloop = content(tensorloop)
        return tensorloop

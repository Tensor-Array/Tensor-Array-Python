from .. import Layer
from .. import Parameter
from tensor_array.core import Tensor
from tensor_array.core import zeros
from tensor_array.core import DataTypes
from typing import Any


class Linear(Layer):
    def __init__(self, bias) -> None:
        """
        Initializes a Linear layer with a specified bias shape.
        Args:
            bias (int): The shape of the bias tensor.
        """
        super().__init__()
        self.bias_shape = bias
        self.b = Parameter(zeros(shape = (bias,), dtype = DataTypes.FLOAT))

    def layer_init(self, t):
        """
        Initializes the layer with the shape of the input tensor.
        Args:
            t (Tensor): The input tensor to determine the shape for the weight parameter.
        """
        self.w = Parameter(zeros(shape = (t[-1], self.bias_shape), dtype = DataTypes.FLOAT))
    
    def calculate(self, t):
        """
        Calculates the linear transformation of the input tensor.
        Args:
            t (Tensor): The input tensor to be transformed.
        Returns:
            Tensor: The transformed tensor after applying the linear transformation.
        """
        return t @ self.w + self.b
        
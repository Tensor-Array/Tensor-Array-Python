"""
# src/tensor_array/core/__init__.py
# This module provides core functionalities for the TensorArray library, including tensor creation, random tensor generation, and data type definitions.
# It includes functions to create tensors filled with zeros or random values, and defines the DataTypes enumeration for various data types supported by the library.
"""

from .tensor import Tensor
from .datatypes import DataTypes

def zeros(shape : tuple, dtype : DataTypes = DataTypes.S_INT_32) -> Tensor:
    """
    Creates a tensor filled with zeros.
    Args:
        shape (tuple): The shape of the tensor.
        dtype (DataTypes): The data type of the tensor.
    Returns:
        Tensor: A tensor filled with zeros.
    """

    from .._ext.tensor2 import zeros as _zeros
    return _zeros(shape, dtype)

def rand(shape : tuple, seed: int = 0) -> Tensor:
    """
    Generates a tensor with random values.
    
    Args:
        shape (tuple): The shape of the tensor.
        seed (int): The seed for random number generation.
    
    Returns:
        Tensor: A tensor filled with random values.
    """

    from .._ext.tensor2 import rand as _rand
    return _rand(shape, seed)

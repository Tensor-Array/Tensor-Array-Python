"""
# src/tensor_array/core/__init__.py
# This module provides core functionalities for the TensorArray library, including tensor creation, random tensor generation, and data type definitions.
# It includes functions to create tensors filled with zeros or random values, and defines the DataTypes enumeration for various data types supported by the library.
"""

from .tensor import Tensor
from .constants import *
from .datatypes import DataTypes
from .operator import *

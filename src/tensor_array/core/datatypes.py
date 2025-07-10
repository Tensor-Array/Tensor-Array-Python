"""
# src/tensor_array/core/datatypes.py
# This module defines an enumeration for various data types supported by the TensorArray library.
# The DataTypes enum includes types such as BOOL, INT, FLOAT, DOUBLE, and others, which correspond to the data types used in tensors.
"""

from .._ext.tensor2 import DataType as _DataType
from enum import Enum

class DataTypes(Enum):
    """
    Enum representing various data types supported by the TensorArray library.
    """
    BOOL = _DataType.BOOL
    S_INT_8 = _DataType.S_INT_8
    S_INT_16 = _DataType.S_INT_16
    S_INT_32 = _DataType.S_INT_32
    S_INT_64 = _DataType.S_INT_64
    FLOAT = _DataType.FLOAT
    DOUBLE = _DataType.DOUBLE
    HALF = _DataType.HALF
    BFLOAT16 = _DataType.BFLOAT16
    U_INT_8 = _DataType.U_INT_8
    U_INT_16 = _DataType.U_INT_16
    U_INT_32 = _DataType.U_INT_32
    U_INT_64 = _DataType.U_INT_64

from .tensor2 import DataType as DataTypeWrapper
from enum import Enum

class DataTypes(Enum):
    BOOL = DataTypeWrapper.BOOL
    S_INT_8 = DataTypeWrapper.S_INT_8
    S_INT_16 = DataTypeWrapper.S_INT_16
    S_INT_32 = DataTypeWrapper.S_INT_32
    S_INT_64 = DataTypeWrapper.S_INT_64
    FLOAT = DataTypeWrapper.FLOAT
    DOUBLE = DataTypeWrapper.DOUBLE
    HALF = DataTypeWrapper.HALF
    BFLOAT16 = DataTypeWrapper.BFLOAT16
    U_INT_8 = DataTypeWrapper.U_INT_8
    U_INT_16 = DataTypeWrapper.U_INT_16
    U_INT_32 = DataTypeWrapper.U_INT_32
    U_INT_64 = DataTypeWrapper.U_INT_64

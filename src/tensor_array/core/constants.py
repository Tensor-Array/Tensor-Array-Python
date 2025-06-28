from .tensor2 import zeros as zerosWrapper
from .tensor import Tensor
from .datatypes import DataTypes

def zeros(shape : Tensor, dtype : DataTypes = DataTypes.S_INT_32):
    return zerosWrapper(shape, dtype)
from .tensor2 import Tensor as TensorWrapper
from .datatypes import DataTypes

class Tensor(TensorWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def transpose(self, dim0: int, dim1: int, isDevive: bool):
        return super().transpose(dim0, dim1, isDevive)
    
    def calc_grad(self):
        super().calc_grad()
    
    def get_grad(self):
        return super().get_grad()
    
    def sin(self):
        return super().sin()
    
    def cos(self):
        return super().cos()
    
    def tan(self):
        return super().tan()

    def sinh(self):
        return super().sinh()
    
    def cosh(self):
        return super().cosh()
    
    def tanh(self):
        return super().tanh()
    
    def log(self):
        return super().log()
    
    def clone(self):
        return super().clone()
    
    def cast(self, dtype: DataTypes):
        return super().cast(dtype)
    
    def numpy(self):
        return super().numpy()
    
    def shape(self):
        return super().shape()
    
    def dtype(self):
        return super().dtype()
    
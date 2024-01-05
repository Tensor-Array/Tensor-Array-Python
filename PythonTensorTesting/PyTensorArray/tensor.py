import numpy as np
import tensor_bind as t

def tensor_decorator(cls):
    return t.TensorC

@tensor_decorator
class Tensor: 
    def __init__(self):
        pass
        

import numpy as np
import tensor as t

class Tensor:
    def __init__(self, arr: np.ndarray):
        self.temp_tensor = t.Tensor(dtype = arr.dtype, shape = arr.shape, data = arr.ctypes.data)





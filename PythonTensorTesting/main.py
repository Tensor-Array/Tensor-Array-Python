# import os

# os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyTensorArray as py_t_arr

import numpy as np

if __name__ == '__main__':
    t3 = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = py_t_arr.Tensor([[1, 2.5, 3], [4, 5, 6]])
    t2 = py_t_arr.Tensor([[1, 2.5], [4, 5], [7, 8]])
    t3 = t1 @ t2
    print("Hello")
    print(t3)

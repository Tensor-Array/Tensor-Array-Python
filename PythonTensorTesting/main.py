# import os

# os.add_dll_directory(os.path.join(os.environ['CUDA_PATH'], 'bin'))

import PyTensorArray as py_t_arr

import numpy as np

if __name__ == '__main__':
    t3 = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = py_t_arr.Tensor(t3)
    print("Hello")
    print(t1.temp_tensor)

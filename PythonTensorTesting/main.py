import PyTensorArray as py_t_arr

import numpy as np

if __name__ == '__main__':
    t4 = np.array([[1, 2, 3], [4, 5, 6]])
    t1 = py_t_arr.tensor.Tensor(t4)
    t2 = py_t_arr.tensor.Tensor(t4)
    t3 = t1 + t2
    print("Hello")
    print(t1)
    print(t2)
    print(t3)
    print(t1 == t2)

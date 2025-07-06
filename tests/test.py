import tensor_array.core as ta
import numpy as np

def test_add():
    example_tensor_array = ta.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.int32))
    example_tensor_array_scalar = ta.Tensor(100)
    example_tensor_sum = example_tensor_array + example_tensor_array_scalar
    print(example_tensor_sum)
    example_tensor_sum.calc_grad()
    print(example_tensor_array.get_grad())
    print(example_tensor_array_scalar.get_grad())

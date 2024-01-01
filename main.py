from lib import tensor2 as t
import numpy as np

print("hello")

t1 = t.Tensor([[1, 2, 3], [4, 5, 6]])
print("tensor len", t1.__len__())
t1 = t1[::, ::2]
print(t1)
t1 = t1.transpose(0, 1)
print("tensor len", t1.__len__())
print(t1)
t2 = t1 + t1
print(t2)
print(t2 > t1)

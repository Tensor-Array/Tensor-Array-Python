from lib import tensor as t
import numpy as np

print("hello")

t1 = t.Tensor([[1], [2]])
t2 = t1 + t1
print(t2)

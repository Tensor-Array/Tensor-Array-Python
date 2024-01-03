from TensorArray.core import tensor2 as t

print("hello")

t1 = t.Tensor([[1, 2, 3], [4, 5, 6]])
t2 = t1.clone()
print("tensor len", t1.__len__())
print(t1)
t1 = t1[::]
print(t1)
t1 = t1.transpose(0, 1)
print("tensor len", t1.__len__())
t4 = t1 @ t2
t3 = t1 * t1
print(t4)
print(t3)

print(t1 != t3)

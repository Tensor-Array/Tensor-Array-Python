from TensorArray.core import tensor2 as t

class Parameter:
    def __init__(self, tensor_param: t.Tensor) -> None:
        self.tensor_param = tensor_param
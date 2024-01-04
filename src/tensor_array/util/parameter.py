from ..core.tensor2 import tensor2 as t

class Parameter:
    def __init__(self, tensor_param: t.Tensor) -> None:
        self.tensor_param = tensor_param

    def update_grad(self):
        self.tensor_param -= self.tensor_param.get_grad().clone()
from .. import Layer
from typing import Any, Callable

class Activation(Layer):
    def __init__(self, activation_function: Callable) -> None:
        super().__init__()
        self.activation_function = activation_function

    def calculate(self, *args: Any, **kwds: Any) -> Any:
        return self.activation_function(*args, **kwds)
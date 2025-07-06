from .. import Layer
from typing import Any, Callable

class Activation(Layer):
    def __init__(self, activation_function: Callable) -> None:
        """
        Initializes an Activation layer with a specified activation function.
        Args:
            activation_function (Callable): The activation function to be applied.
        """
        super().__init__()
        self.activation_function = activation_function

    def calculate(self, *args: Any, **kwds: Any) -> Any:
        """
        Applies the activation function to the input arguments.
        Args:
            *args (Any): Positional arguments to be passed to the activation function.
            **kwds (Any): Keyword arguments to be passed to the activation function.
        Returns:
            Any: The result of applying the activation function to the input arguments.
        """
        return self.activation_function(*args, **kwds)
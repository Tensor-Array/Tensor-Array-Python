"""
# src/tensor_array/core/operator.py
# This module provides various mathematical operations for tensors, including addition, division, multiplication, power, matrix multiplication, and conditional selection.
# Each operation is implemented as a function that takes two tensors as input and returns a new tensor representing the result of the operation.
"""

from .tensor import Tensor

def add(value_1 : Tensor, value_2 : Tensor) -> Tensor:
    """
    Adds two tensors element-wise.
    Args:
        value_1 (Tensor): The first tensor.
        value_2 (Tensor): The second tensor.
    Returns:
        Tensor: A tensor that is the element-wise sum of value_1 and value_2
    """
    from .tensor2 import add as _add
    return _add(value_1, value_2)

def divide(value_1 : Tensor, value_2 : Tensor) -> Tensor:
    """
    Divides two tensors element-wise.
    Args:
        value_1 (Tensor): The first tensor.
        value_2 (Tensor): The second tensor.
    Returns:
        Tensor: A tensor that is the element-wise division of value_1 by value_2
    """
    from .tensor2 import divide as _divide
    return _divide(value_1, value_2)

def multiply(value_1 : Tensor, value_2 : Tensor) -> Tensor:
    """
    Multiplies two tensors element-wise.
    Args:
        value_1 (Tensor): The first tensor.
        value_2 (Tensor): The second tensor.
    Returns:
        Tensor: A tensor that is the element-wise product of value_1 and value_2
    """
    from .tensor2 import multiply as _multiply
    return _multiply(value_1, value_2)

def power(value_1 : Tensor, value_2 : Tensor) -> Tensor:
    """
    Raises the first tensor to the power of the second tensor element-wise.
    Args:
        value_1 (Tensor): The base tensor.
        value_2 (Tensor): The exponent tensor.
    Returns:
        Tensor: A tensor that is the element-wise result of value_1 raised to the power of value_2
    """
    from .tensor2 import power as _power
    return _power(value_1, value_2)

def matmul(value_1 : Tensor, value_2 : Tensor) -> Tensor:
    """
    Performs matrix multiplication between two tensors.
    Args:
        value_1 (Tensor): The first tensor.
        value_2 (Tensor): The second tensor.
    Returns:
        Tensor: A tensor that is the result of matrix multiplication between value_1 and value_2
    """
    from .._ext.tensor2 import matmul as _matmul
    return _matmul(value_1, value_2)

def condition(condition_value : Tensor, value_if_true : Tensor, value_if_false : Tensor) -> Tensor:
    """
    Chooses between two tensors based on a condition tensor.
    Args:
        condition_value (Tensor): The condition tensor.
        value_if_true (Tensor): The tensor to return if the condition is true.
        value_if_false (Tensor): The tensor to return if the condition is false.
    Returns:
        Tensor: A tensor that is either value_if_true or value_if_false, depending on the condition.
    """
    from .._ext.tensor2 import condition as _condition
    return _condition(condition_value, value_if_true, value_if_false)

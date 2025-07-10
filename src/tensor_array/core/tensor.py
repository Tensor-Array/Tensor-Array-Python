"""
# src/tensor_array/core/tensor.py
# This module defines the Tensor class, which represents a multi-dimensional array (tensor) and provides various mathematical operations, shape manipulation, and data type conversion.
# The Tensor class is designed to be used in a computational graph for automatic differentiation.
"""

from .._ext.tensor2 import Tensor as _Tensor
from .datatypes import DataTypes
from __future__ import annotations

class Tensor(_Tensor):
    """
    A class representing a multi-dimensional array (tensor) with various operations.
    This class provides methods for mathematical operations, shape manipulation, and data type conversion.
    It is designed to be used in a computational graph for automatic differentiation.
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the Tensor instance.
        """
        super().__init__(*args, **kwargs)
    
    def transpose(self, dim0: int, dim1: int, isDevive: bool) -> Tensor:
        """
        Transposes the tensor along the specified dimensions.
        Args:
            dim0 (int): The first dimension to transpose.
            dim1 (int): The second dimension to transpose.
            isDevive (bool): Whether to perform the operation in-place on the original tensor.
        Returns:
            Tensor: A new tensor that is the transposed version of the original tensor.
        """
        return super().transpose(dim0, dim1, isDevive)
    
    def calc_grad(self) -> None:
        """
        Calculates the gradient of the tensor with respect to its inputs.
        """
        super().calc_grad()
    
    def get_grad(self) -> Tensor:
        """
        Returns the gradient of the tensor.
        Returns:
            Tensor: A tensor representing the gradient of the original tensor.
        If the tensor does not have a gradient, it returns None.
        """
        return super().get_grad()
    
    def sin(self) -> Tensor:
        """
        Computes the sine of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the sine of each element in the original tensor.
        """
        return super().sin()
    
    def cos(self) -> Tensor:
        """
        Computes the cosine of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the cosine of each element in the original tensor.
        """
        return super().cos()
    
    def tan(self) -> Tensor:
        """
        Computes the tangent of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the tangent of each element in the original tensor.
        """
        return super().tan()

    def sinh(self) -> Tensor:
        """
        Computes the hyperbolic sine of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the hyperbolic sine of each element in the original tensor.
        """
        return super().sinh()
    
    def cosh(self) -> Tensor:
        """
        Computes the hyperbolic cosine of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the hyperbolic cosine of each element in the original tensor.
        """
        return super().cosh()
    
    def tanh(self) -> Tensor:
        """
        Computes the hyperbolic tangent of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the hyperbolic tangent of each element in the original tensor.
        """
        return super().tanh()
    
    def log(self) -> Tensor:
        """
        Computes the natural logarithm of the tensor element-wise.
        Returns:
            Tensor: A tensor containing the natural logarithm of each element in the original tensor.
        """
        return super().log()
    
    def clone(self) -> Tensor:
        """
        Creates a copy of the tensor.
        Returns:
            Tensor: A new tensor that is a copy of the original tensor.
        This method does not perform any operations on the tensor; it simply returns a new instance with the same data.
        """
        return super().clone()
    
    def cast(self, dtype: DataTypes) -> Tensor:
        """
        Casts the tensor to a different data type.
        Args:
            dtype (DataTypes): The target data type to cast the tensor to.
        Returns:
            Tensor: A new tensor that is a copy of the original tensor, but with the specified data type.
        This method does not perform any operations on the tensor; it simply returns a new instance with the same data but in the specified data type.
        """
        return super().cast(dtype)
    
    def numpy(self):
        """
        Converts the tensor to a NumPy array.
        Returns:
            numpy.ndarray: A NumPy array containing the data of the tensor.
        This method allows for easy interoperability with NumPy, enabling the use of NumPy functions and operations on the tensor data.
        Note: This method does not perform any operations on the tensor; it simply returns a NumPy array representation of the tensor.
        """
        return super().numpy()
    
    def shape(self) -> tuple:
        """
        Returns the shape of the tensor.
        Returns:
            tuple: A tuple representing the dimensions of the tensor.
        This method provides the size of each dimension of the tensor, allowing for easy inspection of its structure.
        Note: This method does not perform any operations on the tensor; it simply returns the shape as a tuple.
        """
        return super().shape()
    
    def dtype(self) -> DataTypes:
        """
        Returns the data type of the tensor.
        Returns:
            DataTypes: The data type of the tensor, represented as a DataTypes enum.
        This method provides information about the type of data stored in the tensor, such as whether it is an integer, float, or boolean.
        Note: This method does not perform any operations on the tensor; it simply returns the data type.
        """
        return super().dtype()
    
    def __getitem__(self, item) -> Tensor:
        """
        Gets an item or a slice from the tensor.
        Args:
            item: The index or slice to retrieve.
        Returns:
            Tensor: A new tensor that is a view of the original tensor at the specified index or slice.
        """
        return super().__getitem__(item)
    
    def __len__(self) -> int:
        """
        Returns the number of elements in the first dimension of the tensor.
        Returns:
            int: The size of the first dimension of the tensor.
        """
        return super().__len__()
    
    def __str__(self) -> str:
        """
        Returns a string representation of the tensor.
        Returns:
            str: A string that describes the tensor, including its shape and data type.
        """
        return super().__str__()
    
    def __repr__(self) -> str:
        """
        Returns a detailed string representation of the tensor.
        Returns:
            str: A string that includes the class name, shape, data type, and other relevant information about the tensor.
        """
        return super().__repr__()
    
    def __add__(self, other: Tensor) -> Tensor:
        """
        Adds two tensors element-wise.
        Args:
            other (Tensor): The tensor to add to the current tensor.
        Returns:
            Tensor: A new tensor that is the element-wise sum of the current tensor and the other tensor.
        This method does not modify the original tensors; it returns a new tensor with the result of the addition.
        """
        return super().__add__(other)

    def __sub__(self, other: Tensor) -> Tensor:
        """
        Subtracts two tensors element-wise.
        Args:
            other (Tensor): The tensor to subtract from the current tensor.
        Returns:
            Tensor: A new tensor that is the element-wise difference of the current tensor and the other tensor.
        This method does not modify the original tensors; it returns a new tensor with the result of the subtraction.
        """
        return super().__sub__(other)
    
    def __mul__(self, other: Tensor) -> Tensor:
        """
        Multiplies two tensors element-wise.
        Args:
            other (Tensor): The tensor to multiply with the current tensor.
        Returns:
            Tensor: A new tensor that is the element-wise product of the current tensor and the other tensor.
        This method does not modify the original tensors; it returns a new tensor with the result of the multiplication.
        """
        return super().__mul__(other)
    
    def __truediv__(self, other: Tensor) -> Tensor:
        """
        Divides two tensors element-wise.
        Args:
            other (Tensor): The tensor to divide the current tensor by.
        Returns:
            Tensor: A new tensor that is the element-wise quotient of the current tensor and the other tensor.
        This method does not modify the original tensors; it returns a new tensor with the result of the division.
        """
        return super().__truediv__(other)
    
    def __pow__(self, other: Tensor) -> Tensor:
        """
        Raises the current tensor to the power of another tensor element-wise.
        Args:
            other (Tensor): The tensor representing the exponent.
        Returns:
            Tensor: A new tensor that is the element-wise result of the current tensor raised to the power of the other tensor.
        """
        return super().__pow__(other)
    
    def __matmul__(self, other: Tensor) -> Tensor:
        """
        Performs matrix multiplication between two tensors.
        Args:
            other (Tensor): The tensor to multiply with the current tensor.
        Returns:
            Tensor: A new tensor that is the result of matrix multiplication between the current tensor and the other tensor.
        This method does not modify the original tensors; it returns a new tensor with the result of the matrix multiplication.
        """
        return super().__matmul__(other)

    def __eq__(self, other: Tensor) -> bool:
        """
        Checks if two tensors are equal.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the tensors are equal, False otherwise.
        This method compares the data, shape, and data type of the tensors to determine equality.
        """
        return super().__eq__(other)
    
    def __ne__(self, other: Tensor) -> bool:
        """
        Checks if two tensors are not equal.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the tensors are not equal, False otherwise.
        This method compares the data, shape, and data type of the tensors to determine inequality.
        """
        return super().__ne__(other)
    
    def __lt__(self, other: Tensor) -> bool:
        """
        Checks if the current tensor is less than another tensor.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the current tensor is less than the other tensor, False otherwise.
        """
        return super().__lt__(other)
    
    def __le__(self, other: Tensor) -> bool:
        """
        Checks if the current tensor is less than or equal to another tensor.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the current tensor is less than or equal to the other tensor, False otherwise.
        """
        return super().__le__(other)
    
    def __gt__(self, other: Tensor) -> bool:
        """
        Checks if the current tensor is greater than another tensor.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the current tensor is greater than the other tensor, False otherwise.
        """
        return super().__gt__(other)
    
    def __ge__(self, other: Tensor) -> bool:
        """
        Checks if the current tensor is greater than or equal to another tensor.
        Args:
            other (Tensor): The tensor to compare with the current tensor.
        Returns:
            bool: True if the current tensor is greater than or equal to the other tensor, False otherwise.
        """
        return super().__ge__(other)
    
    def __pos__(self) -> Tensor:
        """
        Returns the tensor itself, unchanged.
        Returns:
            Tensor: The original tensor.
        This method is typically used to indicate that the tensor should be treated as a positive value, but it does not modify the tensor in any way.
        """
        return super().__pos__()
    
    def __neg__(self) -> Tensor:
        """
        Negates the tensor element-wise.
        Returns:
            Tensor: A new tensor with the negated values.
        """
        return super().__neg__()
    
    def __abs__(self) -> Tensor:
        """
        Returns the absolute value of the tensor element-wise.
        Returns:
            Tensor: A new tensor with the absolute values.
        """
        return super().__abs__()
    
    def _hash__(self) -> int:
        """
        Returns a hash value for the tensor.
        Returns:
            int: A hash value representing the tensor.
        This method is useful for using tensors as keys in dictionaries or sets.
        """
        return super()._hash__()

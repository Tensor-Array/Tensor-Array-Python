"""
# src/tensor_array/layers/layer.py
# This module defines the Layer class, which serves as a base class for all layers in the Tensor Array framework.
# The Layer class provides methods for managing parameters, tensors, and other layers,
# as well as methods for initialization and calculation of the layer's output.
"""

from collections import OrderedDict, namedtuple
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing import Any
from tensor_array.core import Tensor
from .parameter import Parameter

class Layer:
    """
    Base class for all layers in the Tensor Array framework.
    This class provides a structure for defining layers, managing parameters, tensors, and other layers.
    It also includes methods for initialization and calculation of the layer's output.
    """
    is_running: bool
    _layers: Dict[str, Optional['Layer']]
    _parameters: Dict[str, Optional[Parameter]]
    _tensors: Dict[str, Optional[Tensor]]

    def __init__(self) -> None:
        """
        Initializes the Layer instance.
        Sets up the initial state of the layer, including whether it is running and initializing empty dictionaries
        for layers, parameters, and tensors.
        """
        super().__setattr__('is_running', False)
        super().__setattr__('_layers', OrderedDict())
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_tensors', OrderedDict())

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """
        Calls the layer with the provided arguments and keyword arguments.
        If the layer is not currently running, it initializes the layer with the shapes of the tensors
        and parameters provided in the arguments and keyword arguments.
        Args:
            *args: Positional arguments, which may include Tensors.
            **kwds: Keyword arguments, which may include Tensors.
        Returns:
            Any: The result of the layer's calculation.
        """
        if not self.__dict__['is_running']:
            list_arg = (t.shape() for t in args if isinstance(t, Tensor))
            dict_kwargs = {
                key: val.shape()
                for key, val in kwds
                if isinstance(val, Tensor)
            }
            self.layer_init(*list_arg, **dict_kwargs)
        super().__setattr__('is_running', True)
        return self.calculate(*args, **kwds)

    def layer_init(self, *args: Tuple, **kwds: Tuple) -> None:
        """
        Initializes the layer with the provided shapes of tensors and parameters.
        This method is called before the layer is run to set up any necessary parameters or tensors.
        Args:
            *args: Positional arguments, which may include shapes of Tensors.
            **kwds: Keyword arguments, which may include shapes of Tensors.
        Returns:
            None
        """
        pass

    def calculate(self, *args: Any, **kwds: Any) -> Any:
        """
        Calculates the output of the layer based on the provided arguments and keyword arguments.
        This method should be overridden in subclasses to implement the specific calculation logic for the layer.
        Args:
            *args: Positional arguments, which may include Tensors.
            **kwds: Keyword arguments, which may include Tensors.
        Returns:
            Any: The result of the layer's calculation.
        """
        pass

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """
        Registers a parameter with the layer.
        This method allows you to add a parameter to the layer, which can be used in calculations
        or for managing the state of the layer.
        Args:
            name (str): The name of the parameter.
            param (Optional[Parameter]): The parameter to register. If None, the parameter is not registered.
        Raises:
            AttributeError: If the method is called before the Layer's __init__ method.
            TypeError: If the name is not a string or if the parameter is not a Parameter instance or None.
            KeyError: If the name contains a dot, is an empty string, or if an attribute with the same name already exists.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string. Got {name}")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        elif not isinstance(param, Parameter) and param is not None:
            raise TypeError(f"cannot assign '{param}' object to parameter '{name}' "
                            "(tensor_array.util.Parameter or None required)")
        else:
            self._parameters[name] = param

    def register_tensor(self, name: str, param: Optional[Tensor]) -> None:
        """
        Registers a tensor with the layer.
        This method allows you to add a tensor to the layer, which can be used in calculations
        or for managing the state of the layer.
        Args:
            name (str): The name of the tensor.
            param (Optional[Tensor]): The tensor to register. If None, the tensor is not registered.
        Raises:
            AttributeError: If the method is called before the Layer's __init__ method.
            TypeError: If the name is not a string or if the parameter is not a Tensor instance or None.
            KeyError: If the name contains a dot, is an empty string, or if an attribute with the same name already exists.
        """
        if '_tensors' not in self.__dict__:
            raise AttributeError("cannot assign tensor before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"tensor name should be a string. Got {name}")
        elif '.' in name:
            raise KeyError("tensor name can't contain \".\"")
        elif name == '':
            raise KeyError("tensor name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._tensors:
            raise KeyError(f"attribute '{name}' already exists")
        elif not isinstance(param, Tensor) and param is not None:
            raise TypeError(f"cannot assign '{param}' object to parameter '{name}' "
                            "(tensor_array.core.tensor2.Tensor or None required)")
        else:
            self._tensors[name] = param

    def register_layer(self, name: str, layer: Optional['Layer']) -> None:
        """
        Registers a layer with the current layer.
        This method allows you to add a sub-layer to the current layer, which can be used
        in calculations or for managing the state of the layer.
        Args:
            name (str): The name of the layer.
            layer (Optional[Layer]): The layer to register. If None, the layer is not registered.
        Raises:
            AttributeError: If the method is called before the Layer's __init__ method.
            TypeError: If the name is not a string or if the layer is not a Layer instance or None.
            KeyError: If the name contains a dot, is an empty string, or if an attribute with the same name already exists.
        """
        if not isinstance(layer, Layer) and layer is not None:
            raise TypeError(f"{layer} is not a Layer subclass")
        elif not isinstance(name, str):
            raise TypeError(f"layer name should be a string. Got {name}")
        elif hasattr(self, name) and name not in self._layers:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"layer name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("layer name can't be empty string \"\"")
        self._layers[name] = layer

    def __setattr__(self, __name: str, __value: Any) -> None:
        """
        Sets an attribute on the layer.
        This method allows you to assign a value to an attribute of the layer.
        If the value is a Parameter, Tensor, or Layer, it registers it appropriately.
        Args:
            __name (str): The name of the attribute to set.
            __value (Any): The value to assign to the attribute.
        Raises:
            TypeError: If the value is not a Parameter, Tensor, Layer, or a valid type for an attribute.
            AttributeError: If the method is called before the Layer's __init__ method.
            KeyError: If the name contains a dot, is an empty string, or if an attribute with the same name already exists.
        """
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if __name in d:
                    if isinstance(d, dict):
                        del d[__name]
                    else:
                        d.discard(__name)
        
        params = self.__dict__.get('_parameters')
        layers = self.__dict__.get('_layers')
        tensors = self.__dict__.get('_tensors')
        if (params is not None and __name in params) or (layers is not None and __name in layers) or (tensors is not None and __name in tensors):
            raise TypeError(f"cannot assign '{__value}' as parameter '{__name}'")
        elif isinstance(__value, Parameter):
            if params is None:
                raise AttributeError("cannot assign parameters before Layer.__init__() call")
            remove_from(self.__dict__, self._layers, self._tensors)
            self.register_parameter(__name, __value)
        elif isinstance(__value, Tensor):
            if layers is None:
                raise AttributeError("cannot assign layers before Layer.__init__() call")
            remove_from(self.__dict__, self._parameters, self._layers)
            self.register_tensor(__name, __value)
        elif isinstance(__value, Layer):
            if tensors is None:
                raise AttributeError("cannot assign layers before Layer.__init__() call")
            remove_from(self.__dict__, self._parameters, self._tensors)
            self.register_layer(__name, __value)
        else:
            super().__setattr__(__name, __value)
    
    def __getattr__(self, __name: str) -> Any:
        """
        Gets an attribute from the layer.
        This method allows you to retrieve an attribute from the layer.
        If the attribute is a registered parameter, tensor, or layer, it returns the corresponding value.
        Args:
            __name (str): The name of the attribute to retrieve.
        Returns:
            Any: The value of the attribute, which can be a Parameter, Tensor, Layer, or any other type.
        Raises:
            AttributeError: If the attribute does not exist in the layer.
        """
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if __name in _parameters:
                return _parameters[__name]
        if '_tensors' in self.__dict__:
            _tensors = self.__dict__['_tensors']
            if __name in _tensors:
                return _tensors[__name]
        if '_layers' in self.__dict__:
            _layers = self.__dict__['_layers']
            if __name in _layers:
                return _layers[__name]
        return super().__getattr__(__name)

    def __delattr__(self, __name: str) -> None:
        """
        Deletes an attribute from the layer.
        This method allows you to remove an attribute from the layer.
        If the attribute is a registered parameter, tensor, or layer, it removes it from the corresponding dictionary.
        Args:
            __name (str): The name of the attribute to delete.
        Raises:
            AttributeError: If the attribute does not exist in the layer.
        """
        if __name in self._parameters:
            del self._parameters[__name]
        elif __name in self._tensors:
            del self._tensors[__name]
        elif __name in self._layers:
            del self._layers[__name]
        else:
            super().__delattr__(__name)

from collections import OrderedDict, namedtuple
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from typing import Any
from ...tensor_array.core import tensor2 as t
from .parameter import Parameter

class Layer:
    _layers = Dict[str, Optional['Layer']]
    _parameters = Dict[str, Optional[Parameter]]
    _tensors = Dict[str, Optional[t.Tensor]]

    def __init__(self) -> None:
        super().__setattr__('_layers', OrderedDict())
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_tensors', OrderedDict())

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
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

    def register_tensor(self, name: str, param: Optional[t.Tensor]) -> None:
        if '_tensors' not in self.__dict__:
            raise AttributeError("cannot assign tensor before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError(f"tensor name should be a string. Got {name}")
        elif '.' in name:
            raise KeyError("tensor name can't contain \".\"")
        elif name == '':
            raise KeyError("tensor name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
        elif not isinstance(param, t.Tensor) and param is not None:
            raise TypeError(f"cannot assign '{param}' object to parameter '{name}' "
                            "(tensor_array.core.tensor2.Tensor or None required)")
        else:
            self._parameters[name] = param

    def register_layer(self, name: str, layer: Optional['Layer']) -> None:
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
        elif isinstance(__value, t.Tensor):
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
        if __name in self._parameters:
            del self._parameters[__name]
        elif __name in self._buffers:
            del self._buffers[__name]
        elif __name in self._modules:
            del self._modules[__name]
        else:
            super().__delattr__(__name)

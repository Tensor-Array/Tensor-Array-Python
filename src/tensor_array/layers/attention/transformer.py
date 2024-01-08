from typing import Any
from .. import Layer
from .attention import MultiheadAttention
from tensor_array.activation import relu
from ..util import Sequential
from ..util import Linear
from ..util import Activation

class TransformerEncoderImpl(Layer):
    def __init__(self, d_model, n_head, ff_size) -> None:
        self.feed_forward = Sequential([
            Linear(ff_size),
            Activation(relu),
            Linear(d_model)
        ])
        self.multihead_attn = MultiheadAttention(d_model, n_head)
        self.layer_norm_1
        self.layer_norm_2

    def calculate(self, input) -> Any:
        attn_output = self.multihead_attn(input, input, input)
        attn_output = self.layer_norm_1(input + attn_output)
        ff_output = self.feed_forward(attn_output)
        return self.layer_norm_2(attn_output + ff_output)
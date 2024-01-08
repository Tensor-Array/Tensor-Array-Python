from typing import Any
from .. import Layer
from ..util import Linear
from tensor_array.core import Tensor
from tensor_array.activation import softmax

def scaled_dot_product_attention(q, k, v, mask = None):
    attn_scores = q @ k.transpose(len(k.shape()) - 2, len(k.shape()) - 1)
    attn_probs = softmax(attn_scores, len(attn_scores.shape()) - 1)
    return attn_probs @ v

class MultiheadAttention(Layer):
    def __init__(self, d_model, n_head) -> None:
        super().__init__()
        self.linear_q = Linear(d_model)
        self.linear_k = Linear(d_model)
        self.linear_v = Linear(d_model)
        self.linear_o = Linear(d_model)
        self.n_head = n_head

    def calculate(self, input_q, input_k, input_v, mask = None) -> Any:
        temp_q = self.linear_q(input_q)
        temp_k = self.linear_k(input_k)
        temp_v = self.linear_v(input_v)

        temp_q = temp_q.reshape((temp_q.shape()[0], temp_q.shape()[1], self.n_head, temp_q.shape()[-1] / self.n_head)).transpose(1, 2)
        temp_k = temp_k.reshape((temp_k.shape()[0], temp_k.shape()[1], self.n_head, temp_k.shape()[-1] / self.n_head)).transpose(1, 2)
        temp_v = temp_v.reshape((temp_v.shape()[0], temp_v.shape()[1], self.n_head, temp_v.shape()[-1] / self.n_head)).transpose(1, 2)

        attention_output = scaled_dot_product_attention(temp_q, temp_k, temp_v, mask)

        attention_output = attention_output.transpose(1, 2)
        attention_output = attention_output.reshape((temp_q.shape()[0], temp_q.shape()[1], temp_q.shape[-2] * temp_q.shape[-1]))
        return self.linear_o(attention_output)

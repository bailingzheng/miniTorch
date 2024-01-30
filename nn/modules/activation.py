import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import Linear
from .. import functional


__all__ = [
    'Hardtanh',
    'LeakyReLU',
    'MultiheadAttention',
    'ReLU',
    'ReLU6',
    'Tanh'
]


class Hardtanh(nn.Module):
    """Applies the HardTanh function element-wise.

    Hardtanh(x) = min(max(min_val, x), max_val)
    """

    def __init__(self, min_val=-1., max_val=1., inplace=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.inplace = inplace
        assert self.max_val > self.min_val

    def forward(self, x):
        return F.hardtanh(x, self.min_val, self.max_val, self.inplace)


class LeakyReLU(nn.Module):
    """Applies the element-wise function:

    LeakyReLU(x) = x if x >= 0; negative_slope * x otherwise.
    """

    def __init__(self, negative_slope=1e-2, inplace=False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, x):
        return F.leaky_relu(x, self.negative_slope, self.inplace)


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information from different representation subspaces 
    as described in the paper: Attention Is All You Need (https://arxiv.org/abs/1706.03762).

    Multi-Head Attention is defined as:
        MultiHead(Q, K, V)=Concat(head_1, ..., head_h) @ W_out

        where head_i = Attention(Q @ W_q, K @ W_k, V @ W_v)

    Parameters
        embed_dim - Total dimension of the model.
        num_heads - Number of parallel attention heads. 
        Note that embed_dim will be split across num_heads (i.e. each head will have dimension embed_dim // num_heads).
    """

    # torch.nn.MultiheadAttention(embed_dim, num_heads, dropout=0.0, bias=True, add_bias_kv=False, 
    #   add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.W_q = Linear(embed_dim, embed_dim)
        self.W_k = Linear(embed_dim, embed_dim)
        self.W_v = Linear(embed_dim, embed_dim)

        self.W_out = Linear(embed_dim, embed_dim)

    # forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, 
    #   average_attn_weights=True, is_causal=False)
    def forward(self, query, key, value, attn_mask=None):
        """
        Parameters
            query (Tensor) - Query embeddings of shape (N, T, E)
            key (Tensor) - Key embeddings of shape (N, S, E)
            value (Tensor) - Value embeddings of shape (N, S, E)

            attn_mask (Optional[Tensor]) - If specified, a 2D or 3D mask preventing attention to certain positions. 
            Must be of shape (T, S) or (N * num_heads, T, S). Binary and float masks are supported. 
            For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            For a float mask, the mask values will be added to the attention weight. 

        Shape
            (N, T, E)[query], (N, S, E)[key], (N, S, E)[value] -> (N, T, E)

            where N is batch size, T is target sequence length, S is source sequence length,
            E is embedding dimension of query and key, and value.

        """

        T, S = query.size(1), value.size(1)

        q = self.W_q(query) # (N, T, E)
        k = self.W_k(key) # (N, S, E)
        v = self.W_v(value) # (N, S, E)

        q = q.view(-1, T, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2)
        k = k.view(-1, S, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) 
        v = v.view(-1, S, self.num_heads, self.embed_dim // self.num_heads).transpose(1, 2) 

        y = functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # y2 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        # print((y2 - y).abs().max())

        y = y.transpose(1, 2).contiguous().view(-1, T, self.embed_dim)
        y = self.W_out(y)
        return y

    
class ReLU(nn.Module):
    """Applies the rectified linear unit function element-wise.

    ReLU(x) = max(0, x)
    """

    # torch.nn.ReLU(inplace=False)
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu(x, inplace=self.inplace)


class ReLU6(Hardtanh):
    """Applies the element-wise function:

    ReLU6(x) = min(max(0, x), 6.)
    """

    def __init__(self, inplace=False):
        super().__init__(min_val=0., max_val=6., inplace=inplace)


class Tanh(nn.Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Shape
        (*) -> (*) where * means any number of dimensions.
    """

    # torch.nn.Tanh(*args, **kwargs)
    def forward(self, x):
        return torch.tanh(x)
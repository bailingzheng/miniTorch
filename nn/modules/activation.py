import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'MultiheadAttention',
    'ReLU',
    'Tanh'
]


class MultiheadAttention(nn.Module):
    """Allows the model to jointly attend to information from different representation subspaces 
    as described in the paper: Attention Is All You Need. (https://arxiv.org/abs/1706.03762)

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

    def _scaled_dot_product_attention(self, query, key, value, attn_mask=None, is_causal=False):
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1))
        attn_bias = torch.zeros(L, S, dtype=query.dtype)
 
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias = attn_bias.masked_fill(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias = attn_bias.masked_fill(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        # attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value, attn_weight


    # forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, 
    #   average_attn_weights=True, is_causal=False)
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        """
        Parameters
            query (Tensor) - Query embeddings of shape (N, L, E)
            key (Tensor) - Key embeddings of shape (N, S, E)
            value (Tensor) - Value embeddings of shape (N, S, E_v)

            attn_mask (Optional[Tensor]) - If specified, a 2D or 3D mask preventing attention to certain positions. 
            Must be of shape (L, S) or (N * num_heads, L, S). Binary and float masks are supported. 
            For a binary mask, a True value indicates that the corresponding position is not allowed to attend. 
            For a float mask, the mask values will be added to the attention weight. 

            is_causal (bool) - If specified, applies a causal mask as attention mask. 

            where N is batch size, T is target sequence length, S is source sequence length,
            E is embedding dimension of query and key, and E_v is embedding dimension of value.

        Shape
            (N, L, E)[query], (N, S, E)[key], (N, S, E_v)[value] -> (N, L, E_v)

        """

        N, L, E = query.shape
        _, S, E_v = value.shape

        q = query.view(N, L, self.num_heads, E // self.num_heads).transpose(1, 2)
        k = key.view(N, S, self.num_heads, E // self.num_heads).transpose(1, 2) 
        v = value.view(N, S, self.num_heads, E_v // self.num_heads).transpose(1, 2) 

        y, attn_weight = self._scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)

        # y2 = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=is_causal)
        # print((y2 - y).abs().max())

        y = y.transpose(1, 2).contiguous().view(N, L, E_v)
        return y, attn_weight


class Tanh(nn.Module):
    """Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

    Shape
        (*) -> (*) where * means any number of dimensions.
    """

    # torch.nn.Tanh(*args, **kwargs)
    def forward(self, x):
        return torch.tanh(x)

    
class ReLU(nn.Module):
    """Applies the rectified linear unit function element-wise.

    ReLU(x) = max(0, x)
    """

    # torch.nn.ReLU(inplace=False)
    def forward(self, x):
        return F.relu(x)
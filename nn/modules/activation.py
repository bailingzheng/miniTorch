
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

    # forward(query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, 
    #   average_attn_weights=True, is_causal=False)
    def forward(self, query, key, value):
        """
        Parameters
            query (Tensor) - Query embeddings of shape (N, T, embed_dim)
            key (Tensor) - Key embeddings of shape (N, T, embed_dim)
            value (Tensor) - Value embeddings of shape (N, T, embed_dim)

            where N is batch size, T is sequence length.

        """

        N, T, _ = query.shape
        H = self.embed_dim // self.num_heads
        tril = torch.tril(torch.ones(T, T)).view(1, 1, T, T)

        q = query.view(N, T, self.num_heads, H).transpose(1, 2) # (N, num_heads, T, H)
        k = key.view(N, T, self.num_heads, H).transpose(1, 2) 
        v = value.view(N, T, self.num_heads, H).transpose(1, 2) 
        
        attn_weights = q @ k.transpose(-2, -1) * k.size(-1)**-0.5 # (N, num_heads, T, T)
        attn_weights = attn_weights.masked_fill(tril[:, :, :T, :T] == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        y = attn_weights @ v # (N, num_heads, T, H)
        y = y.transpose(1, 2).contiguous().view(N, T, self.embed_dim)
        return y, attn_weights


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
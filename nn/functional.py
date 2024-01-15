import math
import torch

def scaled_dot_product_attention(query, key, value, attn_mask=None, is_causal=False):
    T, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1))
    attn_bias = torch.zeros(T, S, dtype=query.dtype)
 
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=0)
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
    return attn_weight @ value
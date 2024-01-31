import torch
import torch.nn as nn


__all__ = [
   'Linear' 
]


class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: y = x @ A.T + b

    Parameters
        in_features - size of each input sample
        out_features - size of each output sample
        bias - If set to False, the layer will not learn an additive bias. Default: True

    Shape
        (*, in_features) -> (*, out_features)
    """

    # torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((in_features, out_features)) / in_features**0.5)
        self.bias = nn.Parameter(torch.randn((1, out_features)) / in_features**0.5) if bias else None

    def forward(self, x):
        y = x @ self.weight
        if self.bias is not None:
            y += self.bias
        return y
# torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
import torch
import torch.nn as nn


__all__ = [
   'Linear' 
]


class Linear(nn.Module):
    """Applies a linear transformation to the incoming data: y = x @ A.T + b

    Parameters
        in_features (int) - size of each input sample
        out_features (int) - size of each output sample
        bias (bool) - If set to False, the layer will not learn an additive bias. Default: True

    Shape
        (*, in_features) -> (*, out_features)
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = torch.randn((in_features, out_features)) / in_features**0.5
        self.bias = torch.zeros((1, out_features)) if bias else None

    def forward(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out

    def parameters(self):
        ps = [self.weight] + ([self.bias] if self.bias is not None else [])
        return ps
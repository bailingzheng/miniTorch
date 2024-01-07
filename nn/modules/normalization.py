import torch
import torch.nn as nn


__all__ = [
    'LayerNorm'
]

class LayerNorm(nn.Module):
    """Applies Layer Normalization over a mini-batch of inputs as described in the paper: 
    Layer Normalization (https://arxiv.org/abs/1607.06450)

    Parameters
        normalized_shape (int or list or torch.Size) - input shape from an expected input of size(*, normalized_shape)
        eps (float) - a value added to the denominator for numerical stability. Default: 1e-5

    Shape
        (N, *) -> (N, *)
    """

    # torch.nn.LayerNorm(normalized_shape, eps=1e-05, elementwise_affine=True, bias=True, device=None, dtype=None)
    def __init__(self, normalized_shape, eps=1e-05):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        dim = -1 if isinstance(self.normalized_shape, int) else list(range(-len(self.normalized_shape), 0))
        xmean = x.mean(dim, keepdim=True)
        xvar = x.var(dim, keepdim=True)
        y = self.gamma * (x - xmean) / (xvar**0.5 + self.eps) + self.beta
        return y
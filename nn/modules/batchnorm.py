import torch
import torch.nn as nn


__all__ = [
    'BatchNorm1d'
]


class BatchNorm1d(nn.Module):
    """Applies Batch Normalization over a 2D or 3D input as described in the paper: 
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 
    (https://arxiv.org/abs/1502.03167)

    Parameters
        num_features (int) - number of features or channels C of the input
        eps (float) - a value added to the denominator for numerical stability. Default: 1e-5
        momentum (float) - the value used for the running_mean and running_var computation. 
            Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1

    Shape
        (*, num_features) -> (*, num_features)
    """

    # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, 
    #   device=None, dtype=None)
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = nn.Parameter(torch.ones((1, num_features)))
        self.beta = nn.Parameter(torch.zeros((1, num_features)))
        self.running_mean = torch.zeros((1, num_features))
        self.running_var = torch.ones((1, num_features))

    def forward(self, x):
        if self.training:
            # expect 2D or 3D input
            dim = 0 if x.dim() == 2 else (0, 1)
            xmean = x.mean(dim, keepdim=True)
            xvar = x.var(dim, keepdim=True)
            y = self.gamma * (x - xmean) / (xvar**0.5 + self.eps) + self.beta
           
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            y = self.gamma * (x - self.running_mean) / (self.running_var**0.5 + self.eps) + self.beta
        return y
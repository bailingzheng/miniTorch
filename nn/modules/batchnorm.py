import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    'BatchNorm1d',
    'BatchNorm2d'
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
        (N, C, *) -> (N, C, *)
        where N is the batch size, C is the number of features or channels.
    """

    # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, 
    #   device=None, dtype=None)
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):

        # expect 2D or 3D input
        dim = 0 
        x_t = x
        if x.dim() == 3:
            dim = (0, 1)
            x_t = x.transpose(1, 2)

        if self.training:
            xmean = x_t.mean(dim)
            xvar = x_t.var(dim, correction=0)
            y_t = self.gamma * (x_t - xmean) / (xvar + self.eps)**0.5 + self.beta

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            y_t = self.gamma * (x_t - self.running_mean) / (self.running_var + self.eps)**0.5 + self.beta

        y = y_t
        if x.dim() == 3:
            y = y_t.transpose(1, 2)

        # y2 = F.batch_norm(x, xmean, xvar, training=True)
        # print((y2 - y).abs().max())
        
        return y


class BatchNorm2d(nn.Module):
    """Applies Batch Normalization over a 4D input as described in the paper: 
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 
    (https://arxiv.org/abs/1502.03167)

    Parameters
        num_features (int) - number of features or channels C of the input
        eps (float) - a value added to the denominator for numerical stability. Default: 1e-5

        momentum (float) - the value used for the running_mean and running_var computation. 
        Can be set to None for cumulative moving average (i.e. simple average). Default: 0.1

    Shape
        (N, C, H, W) -> (N, C, H, W)
        where N is the batch size, C is the number of features or channels.
    """

    # torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, 
    #   device=None, dtype=None)
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.training = True
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x):
        
        # expect 4D input
        x_t = x.transpose(1, 3)
        dim = (0, 1, 2)

        if self.training:
            xmean = x_t.mean(dim)
            xvar = x_t.var(dim, correction=0)
            y_t = self.gamma * (x_t - xmean) / (xvar + self.eps)**0.5 + self.beta

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            y_t = self.gamma * (x_t - self.running_mean) / (self.running_var + self.eps)**0.5 + self.beta

        y = y_t.transpose(1, 3)
        
        # y2 = F.batch_norm(x, xmean, xvar, training=True)
        # print((y2 - y).abs().max())
        
        return y
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        if self.training:
            # expect 2D or 3D input
            if x.dim() == 2:
                dim = 0 
            else:
                dim = (0, 1)
                x = x.transpose(1, 2)
            
            xmean = x.mean(dim)
            xvar = x.var(dim, correction=0)
            y = self.gamma * (x - xmean) / (xvar + self.eps)**0.5 + self.beta
           
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        else:
            y = self.gamma * (x - self.running_mean) / (self.running_var + self.eps)**0.5 + self.beta

        if y.dim() == 3:
            y = y.transpose(1, 2)

        # if x.dim() == 3:
        #     x = x.transpose(1, 2)
        # y_batch_norm = F.batch_norm(x, xmean, xvar, training=True, momentum=self.momentum, eps=self.eps)
        # print((y_batch_norm - y).abs().max())
        
        return y
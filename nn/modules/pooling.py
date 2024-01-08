import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MaxPool2d'
]


class MaxPool2d(nn.Module):
    """Applies a 2D max pooling over an input signal composed of several input planes.

    y[i, j, m, n] = max(x[i, j, S0*m:(K0 + S0*m), S1*n:(K1 + S1*n)])

    where stride is (S0, S1), and kernel size is (K0, K1).

    Parameters
        kernel_size - the size of the window to take a max over
        stride - the stride of the window. Default value is kernel_size

    Shape
        (N, C, H_i, W_i) -> (N, C, H_o, W_o)

        H_o = (H_i - kernel_size[0]) / stride[0] + 1
        W_o = (W_i - kernel_size[1]) / stride[1] + 1

    """

    # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        y = F.max_pool2d(x, self.kernel_size, stride=self.stride)
        return y


    
    
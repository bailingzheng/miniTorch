import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Conv2d'
]

class Conv2d(nn.Module):
    """Applies a 2D convolution over an input signal composed of several input planes.

    y[i, j, m, n] = (weight[j, :, :, :] * x[i, :, (S0*m - P0):(K0 + S0*m - P0), (S1*n - P1):(K1 + S1*n - P1)]).sum() + bias[j]

    where stride is (S0, S1), padding is (P0, P1), and kernel size is (K0, K1).

    Parameters
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0

    Shape
        (N, C_in, H_in, W_in) -> (N, C_out, H_out, W_out)

        H_out = (H_in + 2*padding[0] - kernel_size[0]) / stride[0] + 1
        W_out = (W_in + 2*padding[1] - kernel_size[1]) / stride[1] + 1


    """

    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
    #   bias=True, padding_mode='zeros', device=None, dtype=None)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1])
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]) * k**0.5)
        self.bias = nn.Parameter(torch.rand(out_channels) * k**0.5)
        self.stride = stride
        self.padding = padding

    # Convolution arithmetic: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    def forward(self, x):
        y = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        return y

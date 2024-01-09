import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'Conv2d'
]

class Conv2d(nn.Module):
    """Applies a 2D convolution over an input signal composed of several input planes.

    Parameters
        in_channels (int) - Number of channels in the input image
        out_channels (int) - Number of channels produced by the convolution
        kernel_size (int or tuple) - Size of the convolving kernel
        stride (int or tuple, optional) - Stride of the convolution. Default: 1
        padding (int, tuple or str, optional) - Padding added to all four sides of the input. Default: 0

    Shape
        (N, C_in, H_in, W_in) -> (N, C_out, H_out, W_out)

        H_out = (H_in + 2*P0 - K0) / S0 + 1
        W_out = (W_in + 2*P1 - K1) / S1 + 1

        where stride is (S0, S1), padding is (P0, P1), and kernel size is (K0, K1).
    """

    # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, 
    #   bias=True, padding_mode='zeros', device=None, dtype=None)
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2 
        k = 1.0 / (in_channels * kernel_size[0] * kernel_size[1])
        self.weight = nn.Parameter(torch.rand(out_channels, in_channels, kernel_size[0], kernel_size[1]) * k**0.5)
        self.bias = nn.Parameter(torch.rand(out_channels) * k**0.5)
 

    # Convolution arithmetic: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    def forward(self, x):
        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        N, _, H_in, W_in = x.shape
        padding = [self.padding] * 2 if isinstance(self.padding, int) else self.padding
        kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride

        H_out = (H_in + 2*padding[0] - kernel_size[0]) // stride[0] + 1
        W_out = (W_in + 2*padding[1] - kernel_size[1]) // stride[1] + 1

        w = self.weight.view(self.weight.size(0), -1).t() # (C_in * K0 * K1, C_out)
        x_unf = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride) # (N, C_in * K0 * K1, H_out * W_out)
        y_unf = (x_unf.transpose(1, 2).matmul(w) + self.bias).transpose(1, 2) # (N, C_out, H_out * W_out)
        y = y_unf.view(N, -1, H_out, W_out)

        # y_conv2d = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding)
        # print((y_conv2d - y).abs().max())

        return y

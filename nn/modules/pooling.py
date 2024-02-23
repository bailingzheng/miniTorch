import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'MaxPool2d'
]


class MaxPool2d(nn.Module):
    """Applies a 2D max pooling over an input signal composed of several input planes.

    Parameters
        kernel_size - the size of the window to take a max over
        stride - the stride of the window. Default value is kernel_size

    Shape
        (N, C, H_i, W_i) -> (N, C, H_o, W_o)

        H_out = (H_in + 2*P0 - K0) / S0 + 1
        W_out = (W_in + 2*P1 - K1) / S1 + 1

        where stride is (S0, S1), padding is (P0, P1), and kernel size is (K0, K1).

    """

    # torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride
        self.padding = padding

    def forward(self, x):
        N, C, H_in, W_in = x.shape
        padding = [self.padding] * 2 if isinstance(self.padding, int) else self.padding
        kernel_size = [self.kernel_size] * 2 if isinstance(self.kernel_size, int) else self.kernel_size
        stride = [self.stride] * 2 if isinstance(self.stride, int) else self.stride

        H_out = (H_in + 2*padding[0] - kernel_size[0]) // stride[0] + 1
        W_out = (W_in + 2*padding[1] - kernel_size[1]) // stride[1] + 1

        x_unf = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride) # (N, C * K0 * K1, H_out * W_out)
        x_unf = x_unf.view(N, C, kernel_size[0] * kernel_size[1], H_out * W_out) # (N, C, K0 * K1, H_out * W_out)
        y_unf = x_unf.max(dim=2)[0] # (N, C, H_out * W_out)
        y = y_unf.view(N, C, H_out, W_out)

        # f = F.max_pool2d(x, self.kernel_size, stride=self.stride, padding=self.padding)
        # print(torch.equal(y, f))
        
        return y


    
    
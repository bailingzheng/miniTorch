import torch
import torch.nn as nn

__all__ = [
    'Dropout'
]

class Dropout(nn.Module):
    """During training, randomly zeroes some of the elements of the input tensor with probability p 
    using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

    This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons 
    as described in the paper: Improving neural networks by preventing co-adaptation of feature detectors.
    (https://arxiv.org/abs/1207.0580)

    Furthermore, the outputs are scaled by a factor of 1 / (1 - p) during training. 
    This means that during evaluation the module simply computes an identity function.
    """

    # torch.nn.Dropout(p=0.5, inplace=False)
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x):
        if self.training:
            probs = torch.ones_like(x) * (1 - self.p)
            mask = torch.bernoulli(probs)
            x = x * mask / (1 - self.p)
        return x
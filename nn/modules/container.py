# torch.nn.Sequential(*args: Module)

__all__ = [
    'Sequential'
]

import torch.nn as nn

class Sequential(nn.Module):
    """A sequential container. Modules will be added to it in the order they are passed in the constructor.
    
    The forward() method of Sequential accepts any input and forwards it to the first module it contains. 
    It then “chains” outputs to inputs sequentially for each subsequent module, finally returning the output of the last module.
    """

    def __init__(self, *args):
        super().__init__()
        self.layers = list(args)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        ps = [p for layer in self.layers for p in layer.parameters()]
        return ps
import torch.nn as nn

__all__ = [
    'Flatten'
]

class Flatten(nn.Module):
    """Flattens a contiguous range of dims into a tensor. 

    Parameters
        start_dim - first dim to flatten (default = 1).
        end_dim - last dim to flatten (default = -1).

    Shape
        (*, S_start, ..., S_end, *) -> (*, S_start * ... * S_end, *)
    """

    # torch.nn.Flatten(start_dim=1, end_dim=-1)
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x):
       return x.flatten(self.start_dim, self.end_dim)
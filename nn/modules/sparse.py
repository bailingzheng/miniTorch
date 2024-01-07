import torch
import torch.nn as nn

__all__ = [
    'Embedding'
]

class Embedding(nn.Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size.

    Parameters
        num_embeddings (int) - size of the dictionary of embeddings
        embedding_dim (int) - the size of each embedding vector

    Shape
        (*) -> (*, embedding_dim)
    """

    # torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=None, max_norm=None, 
    #   norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, _freeze=False, device=None, dtype=None)

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = torch.randn(num_embeddings, embedding_dim)

    def forward(self, idx):
        y = self.weight[idx]
        return y

    def parameters(self):
        ps = [self.weight]
        return ps
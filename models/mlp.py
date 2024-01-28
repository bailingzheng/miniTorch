import torch
import torch.nn as tnn

from nn import Embedding, Linear, ReLU

__all__ = [
    'MLP'
]

class MLP(tnn.Module):
    """takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    The architecture is based on the paper: A Neural Probabilistic Language Model
    (https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

    Shape
        (N, S) -> (N, S, V)
        where N is the batch size, S is the block size, and V is the vocabulary size.
    """

    def __init__(self, vocab_size, block_size, num_features):
        super().__init__()
        dim_feedforward = num_features * 4
         
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.wte = Embedding(vocab_size + 1, num_features)
        self.mlp = tnn.Sequential(
            Linear(num_features * block_size, dim_feedforward),
            ReLU(),
            Linear(dim_feedforward, vocab_size)
        )

    def forward(self, idx):
        embs = []

        for _ in range(self.block_size):
            emb = self.wte(idx) # (N, S, E)
            embs.append(emb)

            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.vocab_size

        x = torch.concat(embs, -1) # (N, S, E * S)
        y  = self.mlp(x) # (N, S, V)
        return y
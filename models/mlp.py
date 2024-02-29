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
        hidden_size = num_features * 4
         
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.wte = Embedding(vocab_size + 1, num_features)
        self.mlp = tnn.Sequential(
            Linear(num_features * block_size, hidden_size),
            ReLU()  
        )
        self.lm_head = Linear(hidden_size, vocab_size)

    def forward(self, idx):
        embs = []

        for _ in range(self.block_size):
            emb = self.wte(idx) # (N, S, num_features)
            embs.append(emb)

            idx = torch.roll(idx, shifts=1, dims=1)
            idx[:, 0] = self.vocab_size

        x = torch.concat(embs, dim=-1) # (N, S, num_features * S)
        x = self.mlp(x) # (N, S, hidden_size)
        y = self.lm_head(x)
        return y
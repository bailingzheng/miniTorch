import torch
import torch.nn as tnn
import torch.nn.functional as F

from nn import Embedding, Linear, ReLU
from .lm import LanguageModel


class MLP(LanguageModel):
    """takes the previous block_size tokens, encodes them with a lookup table,
    concatenates the vectors and predicts the next token with an MLP.

    The architecture is based on the paper: A Neural Probabilistic Language Model
    (https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

    Shape
        (N, S) -> (N, S, V)

    """

    def __init__(self, config):
        super().__init__()
        self.S = config.S
        self.V = config.V
        self.wte = Embedding(config.V + 1, config.E)

        self.mlp = tnn.Sequential(
            Linear(config.E * config.S, config.dim_feedforward),
            ReLU(),
            Linear(config.dim_feedforward, config.V)
        )

    def forward(self, idx, target=None):
        embs = []

        for _ in range(self.S):
            emb = self.wte(idx) # (N, S, E)
            embs.append(emb)

            idx = torch.roll(idx, 1, 1)
            idx[:, 0] = self.V

        x = torch.concat(embs, -1) # (N, S, E * S)
        y  = self.mlp(x) # (N, S, V)

        loss = None
        if target is not None:
            loss = F.cross_entropy(y.view(-1, y.size(-1)), target.view(-1), ignore_index=-1)
        
        return y, loss
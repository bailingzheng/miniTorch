
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn as mini_nn

@dataclass
class ModelConfig:
    """Hyperparamters"""
    S: int = None # the sequence length
    V: int = None # the vocabulary size
    E: int = 64 # the feature number

    num_layers: int = 4 # the number of layers
    nhead: int = 4 # the number of heads (in multihead attention)
    dim_feedforward: int = 64 # the dimension of feedforward network


class Bigram(nn.Module):
    """simply a lookup table of logits for the next character given a previous character.

    Shape
        (N, S) -> (N, S, V)
    """

    def __init__(self, config):
        super().__init__()
        V = config.V
        self.bigram = nn.Parameter(torch.zeros((V, V)))

    def forward(self, idx, target):
        y = self.bigram[idx]

        loss = None
        if target is not None:
            loss = F.cross_entropy(y.view(-1, y.size(-1)), target.view(-1), ignore_index=-1)

        return y, loss


class MLP(nn.Module):
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
        self.wte = mini_nn.Embedding(config.V + 1, config.E)

        self.mlp = nn.Sequential(
            mini_nn.Linear(config.E * config.S, config.dim_feedforward),
            mini_nn.ReLU(),
            mini_nn.Linear(config.dim_feedforward, config.V)
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

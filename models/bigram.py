import torch
import torch.nn as nn
import torch.nn.functional as F

from .lm import LanguageModel

__all__ = [
    'Bigram'
]


class Bigram(LanguageModel):
    """simply a lookup table of logits for the next character given a previous character.

    Shape
        (N, S) -> (N, S, V)
    """

    def __init__(self, config):
        super().__init__()
        V = config.V
        self.S = config.S
        self.bigram = nn.Parameter(torch.zeros((V, V)))

    def forward(self, idx, target=None):
        y = self.bigram[idx]

        loss = None
        if target is not None:
            loss = F.cross_entropy(y.view(-1, y.size(-1)), target.view(-1), ignore_index=-1)

        return y, loss
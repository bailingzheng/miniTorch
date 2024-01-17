
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    """Hyperparamters

    block_size - sequence length
    vocab_size - vocabulary size
    n_layer - number of layers
    n_embd - size of character embeddings
    n_head - number of heads (multi head attention)

    """
    block_size: int = None
    vocab_size: int = None
    n_layer: int = 4
    n_embd: int = 64
    n_head: int = 4


class Bigram(nn.Module):
    """simply a lookup table of logits for the next character given a previous character.
    """

    def __init__(self, config):
        super().__init__()
        n = config.vocab_size
        self.logits = nn.Parameter(torch.zeros((n, n)))

    def get_block_size(self):
        return 1

    def forward(self, idx, target):
        y = self.logits[idx]

        loss = None
        if target is not None:
            loss = F.cross_entropy(y.view(-1, y.size(-1)), target.view(-1), ignore_index=-1)

        return y, loss
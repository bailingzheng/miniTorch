import torch
import torch.nn as nn

__all__ = [
    'Bigram'
]


class Bigram(nn.Module):
    """simply a lookup table of logits for the next character given a previous character.

    Shape
        (N, S) -> (N, S, V)
        where N is the batch size, S is the block size, and V is the vocabulary size.
    """

    def __init__(self, vocab_size, block_size):
        super().__init__()
        self.block_size = block_size

        self.bigram = nn.Parameter(torch.zeros((vocab_size, vocab_size)))

    def forward(self, x):
        y = self.bigram[x]
        return y
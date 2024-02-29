import torch
import torch.nn as tnn

from nn import Embedding, Linear, GRUCell, RNNCell

__all__ = [
    'RNN'
]


class RNN(tnn.Module):
    """RNN language model: either a vanilla RNN or a GRU.

    Shape
        (N, S) -> (N, S, V)
        where N is the batch size, S is the block size, and V is the vocabulary size.
    """

    def __init__(self, vocab_size, block_size, num_features, cell_type='gru'):
        super().__init__()
        hidden_size = num_features * 4

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.h0 = tnn.Parameter(torch.zeros(1, hidden_size)) 
        self.wte = Embedding(vocab_size, num_features)

        if cell_type == 'gru':
            self.cell = GRUCell(num_features, hidden_size)
        else:
            self.cell = RNNCell(num_features, hidden_size)

        self.lm_head = Linear(hidden_size, vocab_size)

    def forward(self, x):
        N, S = x.size()
        x = self.wte(x) # (N, S, num_features)

        h_prev = self.h0.expand((N, -1))
        hiddens = []
        for t in range(S):
            xt = x[:, t, :] # (N, num_features)
            h = self.cell(xt, h_prev) #(N, hidden_size)
            h_prev = h
            hiddens.append(h)

        y = self.lm_head(torch.stack(hiddens, dim=1))
        return y
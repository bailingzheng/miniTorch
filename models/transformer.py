import torch
import torch.nn as tnn

from nn import Embedding, TransformerEncoderLayer, TransformerEncoder, LayerNorm, Linear

__all__ = [
    'Transformer'
]

class Transformer(tnn.Module):
    """Transformer Language Model, exactly as seen in GPT-2

    Shape
        (N, S) -> (N, S, V)
        where N is the batch size, S is the block size, and V is the vocabulary size.
    """

    def __init__(self, vocab_size, block_size, num_features):
        super().__init__()
        nhead = 4
        dim_feedforward = num_features * 4
        num_layers = 4

        self.block_size = block_size
        self.wte = Embedding(vocab_size, num_features)
        self.wpe = Embedding(block_size, num_features)

        encoder_layer = TransformerEncoderLayer(num_features, nhead, dim_feedforward=dim_feedforward, dropout=0.0)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        self.ln = LayerNorm(num_features)
        self.lm_head = Linear(num_features, vocab_size)

    def forward(self, x):
        _, S = x.shape
        attn_mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)
        pos_emb = self.wpe(torch.arange(S, dtype=torch.long)).unsqueeze(0) # (1, S, num_features)

        x = self.wte(x) # (N, S, num_features)
        x = x + pos_emb
        x = self.encoder(x, mask=attn_mask)
        x = self.ln(x)
        y = self.lm_head(x)
        return y
import torch
import torch.nn.functional as F

from nn import Embedding, TransformerEncoderLayer, TransformerEncoder, LayerNorm, Linear
from .lm import LanguageModel

__all__ = [
    'Transformer'
]

class Transformer(LanguageModel):
    """Transformer Language Model, exactly as seen in GPT-2

    Shape
        (N, S) -> (N, S, V)
    
    """

    def __init__(self, config):
        super().__init__()
        self.S = config.S
        self.wte = Embedding(config.V, config.E)
        self.wpe = Embedding(config.S, config.E)

        encoder_layer = TransformerEncoderLayer(config.E, config.nhead, dim_feedforward=config.dim_feedforward, dropout=0.0)
        self.encoder = TransformerEncoder(encoder_layer, config.num_layers)
        self.ln = LayerNorm(config.E)
        self.lm_head = Linear(config.E, config.V)

    def forward(self, idx, target=None):
        _, S = idx.shape
        attn_mask = torch.triu(torch.full((S, S), float('-inf')), diagonal=1)

        tok_emb = self.wte(idx) # (N, S, E)
        pos_emb = self.wpe(torch.arange(S, dtype=torch.long)).unsqueeze(0) # (1, S, E)
        x = tok_emb + pos_emb
        y = self.lm_head(self.ln(self.encoder(x, mask=attn_mask)))

        loss = None
        if target is not None:
            loss = F.cross_entropy(y.view(-1, y.size(-1)), target.view(-1), ignore_index=-1)
        
        return y, loss
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
    dim_feedforward: int = E * 4 # the dimension of feedforward network


class LanguageModel(nn.Module):

    @torch.no_grad()
    def generate(self, idx, temperature=1.0, do_sample=False, top_k=None):
        """Take a conditioning sequence of indices idx (LongTensor of shape (N, S)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.

        """
        for _ in range(self.S):

            logits, _ = self.forward(idx)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


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


class Transformer(LanguageModel):
    """Transformer Language Model, exactly as seen in GPT-2

    Shape
        (N, S) -> (N, S, V)
    
    """

    def __init__(self, config):
        super().__init__()
        self.S = config.S
        self.wte = mini_nn.Embedding(config.V, config.E)
        self.wpe = mini_nn.Embedding(config.S, config.E)

        encoder_layer = mini_nn.TransformerEncoderLayer(config.E, config.nhead, dim_feedforward=config.dim_feedforward, dropout=0.0)
        self.encoder = mini_nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.ln = mini_nn.LayerNorm(config.E)
        self.lm_head = mini_nn.Linear(config.E, config.V)

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
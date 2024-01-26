from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ModelConfig',
    'LanguageModel'
]


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
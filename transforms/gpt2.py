import json
from functools import lru_cache

import regex as re
import torch.nn as tnn

__all__ = [
    "GPT2BPETokenizer"
]

@lru_cache()
def bytes_to_unicode():
    """Returns list of utf-8 byte and a corresponding list of unicode strings."""

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """Return set of symbol pairs in a word."""
    
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2BPETokenizer(tnn.Module):
    """Transform for GPT-2 BPE Tokenizer.
    Original openai implementation https://github.com/openai/gpt-2/blob/master/src/encoder.py

    Parameters:
        encoder_json_path - Path to GPT-2 BPE encoder json file.
        vocab_bpe_path - Path to bpe vocab file.
    """

    def __init__(self, encoder_json_path, vocab_bpe_path):
        super().__init__()

        with open(encoder_json_path, "r", encoding="utf-8") as f:
            self.encoder = json.load(f)

        with open(vocab_bpe_path, "r", encoding="utf-8") as f:
            vocab = f.read()
            bpe_merges = [tuple(line.split()) for line in vocab.split("\n")[1:-1]]

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.decoder = {v:k for k, v in self.encoder.items()}
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}

    def _bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float("inf")))

            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            new_word = tuple(new_word)
            word = new_word

            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        
        word = " ".join(word)
        self.cache[token] = word
        return word

    def forward(self, text):
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self._bpe(token).split(" "))

        return bpe_tokens

    def decode(self, tokens):
        str = "".join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in str]).decode("utf-8", errors="replace")
        return text
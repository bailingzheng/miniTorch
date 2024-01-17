import torch
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader


class CharDataset(Dataset):

    def __init__(self, words, chars, max_word_length):
        self.words = words
        self.chars = chars
        self.max_word_length = max_word_length

        self.stoi = {ch:i+1 for i, ch in enumerate(chars)}
        self.itos = {i:ch for ch, i in self.stoi.items()}

    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1

    def get_output_length(self):
        return self.max_word_length + 1

    def encode(self, word):
        return torch.tensor([self.stoi[ch] for ch in word], dtype=torch.long)

    def decode(self, ix):
        return ''.join([self.itos[i] for i in ix])

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)

        x = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        y = torch.zeros(self.max_word_length + 1, dtype=torch.long)
        
        x[1:1+len(ix)] = ix
        y[:len(ix)] = ix
        y[len(ix)+1:] = -1 # index -1 will mask the loss at the inactive locations

        return x, y


class InfiniteDataLoader:

    def __init__(self, dataset, **kwargs):
        train_sampler = RandomSampler(dataset, replacement=True, num_samples=int(1e10))
        self.train_loader = DataLoader(dataset, sampler=train_sampler, **kwargs)
        self.data_iter = iter(self.train_loader)

    def next(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_loader)
            batch = next(self.data_iter)
        
        return batch
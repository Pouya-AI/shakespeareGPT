import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, block_size, n_emb, head_size):
        super(Head, self).__init__()

        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        print(f'{x.shape = }')
        k = self.key(x)
        print(f'{k.shape = }')
        q = self.query(x)
        print(f'{q.shape = }')
        v = self.value(x)
        print(f'{v.shape = }')
        wei = k @ q.permute(0, -1, -2) * C ** -0.5
        print(f'{self.tril.shape = }')
        wei = wei.masked_fill(self.tril[:T,:T] == 0, value=float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out






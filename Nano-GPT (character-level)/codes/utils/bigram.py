import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# BigramLanguageModel class
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(BigramLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        self.logits = self.token_embedding_table(idx)  # (B,T,S)

        if targets == None:
            loss = None
        else:
            B, T, C = self.logits.shape
            self.logits = self.logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(self.logits, targets)

        return self.logits, loss

    def generate(self, idx, max_new_tokens):

        for i in range(max_new_tokens):
            # get the prediction
            logits, loss = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1).squeeze(1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sample index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx



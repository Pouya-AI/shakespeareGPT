import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.loader import GetBatch
from utils.utils import guess_loss
# from utils.bigram import BigramLanguageModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# to download tiny shakespeare file please run wget command in terminal
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
file_name = 'input.txt'
file = open(file_name)
text = file.read()
chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))

# lets now split up the data into train and validation sets
# (90% --> train / 10% --> test)

n = int(0.9*len(data))
data_train = data[:n]
data_test = data[n:]


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



loader_train = GetBatch(data_train, 32, 8)
loader_test = GetBatch(data_test, 32, 8)

model = BigramLanguageModel(vocab_size).to(device)
loss_train = guess_loss(model,loader_train)
print(f'{loss_train = }')
loss_test = guess_loss(model,loader_test)
print(f'{loss_test = }\n')

idx = torch.tensor([[0]]).to(device)
text_gen = model.generate(idx,100)
text_gen = text_gen.to('cpu')
print(f'Generated text before optimization --> {decode(text_gen[0].tolist())}\n')


optimizer = optim.Adam(model.parameters(),lr=0.01)
schedular = optim.lr_scheduler.ExponentialLR(optimizer,0.7)

epochs = 7000
for epoch in range(epochs):
    xb,yb = loader_train.get_batch()
    xb = xb.to(device)
    yb = yb.to(device)
    logits,loss = model(xb,yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch%1000 == 0:
        schedular.step()
        print(loss.item())


loss_train = guess_loss(model,loader_train)
print(f'\n{loss_train = }')
loss_test = guess_loss(model,loader_test)
print(f'{loss_test = }\n')

idx = torch.tensor([[0]]).to(device)
text_gen = model.generate(idx,100)
text_gen = text_gen.to('cpu')
print(f'Generated text after optimization --> {decode(text_gen[0].tolist())}\n')

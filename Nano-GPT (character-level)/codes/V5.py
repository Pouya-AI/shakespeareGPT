# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# device = 'cuda' if torch.cuda.is_available() == True else 'cpu'
device = 'cpu'
print(f'device: {device}')


# Class GetBatch : loader
class GetBatch:
    def __init__(self, data, batch_size, block_size):
        self.data = data
        self.batch_size = batch_size
        self.block_size = block_size

    def get_batch(self):
        # generate a small batch of data of input x and targets 7
        # get_batch serves as a dataloader
        ix = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        x = torch.stack([self.data[i:i + self.block_size] for i in ix], dim=0)
        y = torch.stack([self.data[i + 1:i + self.block_size + 1] for i in ix], dim=0)
        return x, y


# Function guess_loss
@torch.no_grad()
def guess_loss(model, loader, eval_iters=None):
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    if eval_iters is None:
        eval_iters = 100
    losses = torch.zeros(eval_iters, dtype=torch.float32).to(device)
    for k in range(eval_iters):
        xb, yb = loader.get_batch()
        xb = xb.to(device)
        yb = yb.to(device)
        yb = yb.flatten(end_dim=1)
        logits = model(xb, yb)
        logits = logits.flatten(end_dim=1)
        loss = loss_fn(logits, yb)
        losses[k] = loss.item()
    avg_loss = losses.mean().item()
    model.train()
    return avg_loss


# Create Loaders and datasets
file_name = 'alllines.txt'
# file_name = '/kaggle/input/tiny-shakespeare/tiny-shakespeare.txt'
file = open(file_name)
text = file.read()
text = text.replace('"', '')

chars = sorted(set(list(text)))
vocab_size = len(chars)

stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for s, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))

n = int(0.9 * len(data))
data_train = data[:n]
data_test = data[n:]
batch_size = 32
block_size = 32
loader_train = GetBatch(data_train, batch_size, block_size)
loader_test = GetBatch(data_test, batch_size, block_size)


# Class Head
class Head(nn.Module):
    def __init__(self, block_size, n_emb, head_size=None):
        super(Head, self).__init__()

        if head_size is None:
            head_size = n_emb

        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device))

    #         self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = k @ q.permute(0, -1, -2) * C ** -0.5
        #         wei = self.dropout(wei)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, value=float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out


# Class MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, block_size, n_emb, head_size=None):
        super(MultiHeadAttention, self).__init__()
        if head_size is None:
            head_size = 32
        self.heads = nn.ModuleList([Head(block_size, n_emb, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, num_heads * head_size)

    #         self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        #         out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, num_f):
        super(FeedForward, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_features=num_f, out_features=4 * num_f),
            nn.ReLU(),
            nn.Linear(in_features=4 * num_f, out_features=num_f),
            #             nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.net(x)
        return out


# FeedForward Class
class Block(nn.Module):
    """ a simple linear model followed by a non_linearity """

    def __init__(self, block_size, num_heads, head_size, n_emb):
        super(Block, self).__init__()
        self.sa_head = MultiHeadAttention(num_heads=num_heads,
                                          block_size=block_size, n_emb=n_emb, head_size=head_size)
        mlp_f = num_heads * head_size
        self.ffwd = FeedForward(mlp_f)
        self.ln1 = nn.LayerNorm(num_heads * head_size)
        self.ln2 = nn.LayerNorm(num_heads * head_size)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        out = x + self.ffwd(self.ln2(x))
        return out


class BigramLanguageModel(nn.Module):
    def __init__(self, block_size, vocab_size, n_emb, num_heads, head_size):
        super(BigramLanguageModel, self).__init__()
        self.block_size = block_size
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.position_embedding_table = nn.Embedding(block_size, n_emb)
        self.blocks = nn.Sequential(
            Block(block_size, num_heads, head_size, n_emb),
            Block(block_size, num_heads, head_size, num_heads * head_size),
            Block(block_size, num_heads, head_size, num_heads * head_size),
            nn.LayerNorm(num_heads * head_size)
        )

        self.lm_head = nn.Linear(num_heads * head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -block_size:]
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,S)
        pos_emb = self.position_embedding_table(torch.arange(self.block_size, device=device))
        x = tok_emb + pos_emb[:T]
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens=None):

        if max_new_tokens is None:
            max_new_tokens = 100
        for i in range(max_new_tokens):
            logits = self(idx)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs[:, -1, :], num_samples=1, replacement=True)
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx


dropout = 0.1
bigram = BigramLanguageModel(block_size=block_size, vocab_size=vocab_size, n_emb=384, num_heads=6, head_size=64).to(
    device)
print(f"bigram's device: {device}")

xb, yb = loader_train.get_batch()
xb = xb.to(device)
yb = yb.to(device)
out = bigram(xb)
idx = torch.tensor([[1]], device=device)
idx_gen = bigram.generate(idx, max_new_tokens=300)
print(f'''Generated text before optimization --> \n{decode(idx_gen[0].tolist())[1:]}''')

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(bigram.parameters(), lr=0.009)
schedular = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.93)

# optimizer.state_dict()['param_groups'][0]['lr']

bigram.train()
num_epochs = 1000
for epoch in range(num_epochs):
    xb, yb = loader_train.get_batch()
    xb = xb.to(device)
    yb = yb.to(device)
    yb = yb.flatten(end_dim=1)
    logits = bigram(xb)
    logits = logits.flatten(end_dim=1)
    loss = loss_fn(logits, yb)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 0:
        print(f'''epoch {epoch}: loss = {loss.item()}''')
        schedular.step()

with torch.no_grad():
    bigram.eval()
    idx = torch.tensor([[0]], device=device)
    idx_gen = bigram.generate(idx, max_new_tokens=2000)
    print(f'''Generated text after optimization --> \n{decode(idx_gen[0].tolist())[1:]}''')

loss_train = guess_loss(bigram, loader_train, eval_iters=100)
print(f'loss_train = {loss_train}')
loss_test = guess_loss(bigram, loader_test, eval_iters=100)
print(f'loss_test = {loss_test}')

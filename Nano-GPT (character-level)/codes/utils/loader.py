import torch


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



# file_name = 'input.txt'
# file = open(file_name)
# text = file.read()
# chars = sorted(set(list(text)))
# vocab_size = len(chars)
#
# stoi = {s: i for i, s in enumerate(chars)}
# itos = {i: s for s, i in stoi.items()}
# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: ''.join([itos[i] for i in l])
#
# data = torch.tensor(encode(text))
# loader = GetBatch(data, 4, 8)
#
# x, y = loader.get_batch()
# print(x)
# print(y)

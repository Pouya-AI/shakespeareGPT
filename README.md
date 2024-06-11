building & training GPT from scratch based on [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out. tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)

dataset: Shakespeare plays from kaggle (https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays?select=alllines.txt) with slight modification.

## character-level Nano GPT (token level is comming soon).
number of parameters: 5478221 (~5.5 million)
block_size = 256
vocab_size = 77
embedding dimesion: 384
number of attention heads: 6
each attention head size: 64 (384/6)
I used no dropout layers 

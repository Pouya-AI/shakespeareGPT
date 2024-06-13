class Tokenizer_Base:
    def get_stats(self,ids):
        counts = {}
        for i1,i2 in zip(ids,ids[1:]):
            if i1 == 256 or i2 == 256:
                continue
            counts[(i1,i2)] = counts.get((i1,i2),0) + 1
        return counts
    
    def merge(self,tokens,pair,idx):
        new_tokens = []
        i=0
        while i<(len(tokens)):
            if i<len(tokens)-1 and tokens[i] == pair[0] and  tokens[i+1] == pair[1]:
                new_tokens.extend([idx])
                i+=1
            else:
                new_tokens.extend([tokens[i]])
            i+=1
        return new_tokens
    
    def get_vocabs(self,merges):
        vocabs = {idx:bytes([idx]) for idx in range(256)}
        for pair,idx in merges:
            vocabs[idx] = vocabs[pair[0]] + vocabs[pair[1]]
        return vocabs
    
    def regex_tokenizer(self,text,pattern=None):
        import regex as re
        if pattern == None:
            # GPT4 pattern
            pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
        text_sep = re.findall(pattern,text)
        tokens_sep = [list(piece.encode()) for piece in text_sep]
        ids = []
        for sep in [sep+[256] for sep in tokens_sep]:
            ids.extend(sep)
        return ids
    
    def save(self,merges,vocabs):
        import pickle
        with open('merges.pkl','wb') as f:
            pickle.dump(merges, f)
        with open('vocabs.pkl','wb') as f:
            pickle.dump(vocabs, f)
    
    def load(self,file_name):
        import pickle
        try:
            with open(f'{file_name}','rb') as file:
                vocabs = pickle.load(file)
            return vocabs 
        except:
            raise FileNotFoundError(f'{file_name} does not exist. please train tokenizer or prepare the file')           
            
            
            
            
class Tokenizer(Tokenizer_Base):
    def __init__(self,merges=None,vocabs=None):
        self.vocabs = vocabs
        self.merges = merges
        
    def encode(self,text):
        if self.merges == None:
            self.merges = self.load('merges.pkl')
        
        merges = dict(self.merges)
        ids = list(text.encode())
        while 1:
            pair = min([(t0,t1) for t0,t1 in zip(ids,ids[1:])],key=lambda x: merges.get(x,float('inf')))
            if merges.get(pair) == None:
                break
            ids = self.merge(ids,pair,merges.get(pair))
        return ids
        
    def decode(self,tokens):
        if self.vocabs == None:
            self.vocabs = self.load('vocabs.pkl')
        
        return ''.join([self.vocabs[idx].decode(errors='replace') for idx in tokens])
        
    def tokenize(self,text):
        tokens = self.encode(text)
        return [self.decode([token]) for token in tokens]
            
    
    def train(self,text,num_vocabs=None):
        ids = self.regex_tokenizer(text)
        self.merges = []
        if num_vocabs == None:
            i = 0
            while 1:
                stats = self.get_stats(ids)
                if len(stats)==0:
                    break
                if i%1000 == 0:
                    print(f'merge {i} --> {len(ids) = }')
                top_pair = max(stats.items(),key=lambda x: x[1])[0]
                idx = 257+i
                self.merges.append((top_pair,idx))
                ids = self.merge(ids,top_pair,idx)
                i += 1
        else:
            assert num_vocabs >= 0
            num_merges = num_vocabs - 256
            from tqdm import tqdm
            for i in tqdm(range(num_merges)):
                stats = self.get_stats(ids)
                if len(stats)==0:
                    print("Vocab size is too large. There is no more mergable tokens")
                    break
                top_pair = max(stats.items(),key=lambda x: x[1])[0]
                idx = 257+i
                self.merges.append((top_pair,idx))
                ids = self.merge(ids,top_pair,idx)
                
        self.vocabs = self.get_vocabs(self.merges)
        return ids,self.merges,self.vocabs

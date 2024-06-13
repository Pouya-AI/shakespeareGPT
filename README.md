# Nano GPT character level
## Review
building & training GPT from scratch based on [Andrej Karpathy: Let's build GPT: from scratch, in code, spelled out. tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## Dataset
Shakespeare plays from kaggle (https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays?select=alllines.txt) with slight modification.

## Model 
### character-level Nano GPT (token level is comming soon).
number of parameters: 5478221 (~5.5 million)

block_size = 256

vocab_size = 77 

embedding dimesion: 384

number of attention heads: 6

each attention head size: 64 (384/6)

no dropout layers 

more specification in [Nano-GPT (character-level)/codes/notebooks]

## Artifacts
### Generated text before optimization

"TxJqIi4we!KZTm	SwqzXWGEaThiQwupjYLYAluo,5wVG7[ x3S	qNu'o5mE,	K3'0Xz:84OgToP,:[smI$r8m  
),glzH  
s3Ruvy89DmUz9zkwiNMuK)cp	PhT5RLjG2sK  
UsIj:A UzuHod3aKQ:tp7ygVjAN]azCOT9n6D'fLCLz,]pYB9N.$ 1[pJ,CoJJBgQJ	'1GncADl9DuuoOK7!mywd':]SqeiK!W?Y.78jK(0sQXT  
oNIMFLAKg  
j!0XT)mP	)NgTgr9w,:LC6.9]t9R2p]O6j491X97A0T 0b  
["  

### Generated text after optimization
Where's hath show'd with createst sell.  
When they are before, but to-day?  
Now can I not of her learned in you befriends?  
He is a few in us in honey-house, about of awn,  
To see wind the offer, I took you.  
I hear further he chair  
He best, is an endure unto successity  
A tables? Tribunes in Quarvellous in,  
And not mark the king! What our dear man?  
To-morrow, you hear that has boy, careers,  
God, first is out or to strong upon him.  
SILV  
I have lady, too cry that you hurtle: sure, must  
If any all first infirmitie, and to your majesty,  
I saw here is more profits fair nothing rough,  
Do we willingly work.  
The was good disputal bought captain shall be,  
Out only Guildens there comes ere words,  
That seek up.  
Who's Murder doth Antony is too her: the face?  
I Aleveller, of the horship content:  
The more more was where worth that must fortunes,  
Henry all the portiverance honourable,  
This spirit and shield not Timon:  
And hath by the King on, the letter'd it on  
Shall be true convenitied the less me.  
They be cried the ran with that Christian sod glory  
Who washes, and be those wondrous else eyes,  
As good on that vengeance and batt than great.  
Lay dhy being that I have laid from Troy  
How o'er I, with thee a dove a most afoot  
But, on thyself at office, from five her  
Must dovers, an evil think I will make perforce.  

# Nano GPT word level 
## Review
Building and training shakespeare GPT (token level) from scrach.  
I used the cl100k_base regex pattern, the tokenizer for GPT-4, to build my tokenizer.  
Negative Sampling algorithm is used to obtain embeddings for shakespeare's tokens. The only difference is that I used both the next token and the preceding token relative to the context, rather than just the next token, in Negative Sampling.  

## Dataset
shakespeare's text book 

## Model
not implemented yet 

from typing import List, Tuple, Dict, Set, Union, Any, cast, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from collections import Counter
import os
import pickle


class SlidingWindowDataset(Dataset):
    def __init__(self, stories: list, params: dict):
        super(SlidingWindowDataset, self).__init__()
        print('Building dataset...')
        self.seq_len = params['seq_len']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer.model_max_length = 5000
        if len(stories) < 300_000:
            self.stories = ' '.join([s for s in stories[:1000] if s != ''])
        else:
            self.stories = ' '.join([s for s in stories[:400_000]]) #100_000:300_000
        self.tokenized_data = self.tokenizer.encode(self.stories, add_special_tokens=False)   
        print(f'Done building dataset with {len(self.tokenized_data):,} tokens')     
    def __len__(self) -> int:
        return len(self.tokenized_data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.tokenized_data[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.tokenized_data[idx + self.seq_len], dtype=torch.long)
        return x, y

class MyModel(nn.Module):
    def __init__(self, params: dict):
        super(MyModel, self).__init__()
        self.params = params

        self.embeddings = nn.Embedding(
            params['vocab_size'],
            params['embed_size']
        )
        self.pe = self._generate_positional_encodings().to(params['device']) # (seq_len, embed_size) pe is not a param so must be sent to device manually
        self.transformers = nn.ModuleList([
            Transformer(params) for _ in range(params['num_transformers'])
        ])
        self.ff = nn.Sequential(
            nn.Linear(params['embed_size'], params['vocab_size']),
        )
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Model init with device: ', params['device'])
        #print total params with 000, formatting
        print(f'Total params: {total_params:,}')

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size = self.params['batch_size']
        seq_len = self.params['seq_len']
        embed_size = self.params['embed_size']
        vocab_size = self.params['vocab_size']

        # x: (batch_size, seq_len)
        x = self.embeddings(x) # x: (batch_size, seq_len, embed_size)
        # mask = mask.unsqueeze(-1)
        # mask = mask.expand_as(x) # mask: (batch_size, seq_len, embed_size)
        x += self.pe
        assert x.shape == (batch_size, seq_len, embed_size), f'failed x shape; {x.shape}, and should be {(batch_size, seq_len, embed_size)}'

        # transformers
        for transformer in self.transformers:
            x = transformer(x)
        assert x.shape == (batch_size, seq_len, embed_size), f'failed x shape; {x.shape}, and should be {(batch_size, seq_len, embed_size)}'
        
        # projection to vocab size
        x = self.ff(x)
        #pooling (batch size, vocab size)
        x = torch.mean(x, dim=1)

        assert x.shape == (batch_size, vocab_size), f'failed x shape; {x.shape}, and should be {(batch_size, vocab_size)}'
        return x

    def _generate_positional_encodings(self):
        seq_len = self.params['seq_len']
        embed_size = self.params['embed_size']
        # Initialize positional encoding matrix
        pos_encodings = torch.zeros(seq_len, embed_size)
        # Generate encoding for each position
        for pos in range(seq_len):
            for i in range(0, embed_size, 2):
                pos_encodings[pos, i] = math.sin(pos / 10000 ** (i / embed_size))
                if i + 1 < embed_size:  # Ensure index is in range
                    pos_encodings[pos, i + 1] = math.cos(pos / 10000 ** ((i + 1) / embed_size))
        assert pos_encodings.shape == (seq_len, embed_size), f'failed in positional encoding shape {pos_encodings.shape}, and should be {(seq_len, embed_size)}'
        return pos_encodings

class Transformer(nn.Module):
    def __init__(self, params: dict):
        super(Transformer, self).__init__()
        self.params = params
        
        # attention
        self.q_w = nn.Parameter(torch.randn(params['embed_size'], params['embed_size']))
        self.k_w = nn.Parameter(torch.randn(params['embed_size'], params['embed_size']))
        self.v_w = nn.Parameter(torch.randn(params['embed_size'], params['embed_size']))
        self.linear_out = nn.Linear(params['embed_size'], params['embed_size'])
        self.ln = nn.LayerNorm(params['embed_size'])

        # feedforward network
        self.ff = nn.Sequential(
            nn.Linear(params['embed_size'], params['hidden_size']),
            nn.GELU(),
            nn.Linear(params['hidden_size'], params['embed_size'])
        )
        self.ln2 = nn.LayerNorm(params['embed_size'])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = self.params['batch_size']
        seq_len = self.params['seq_len']
        embed_size = self.params['embed_size']
        n_heads = self.params['n_heads']
        head_dim = embed_size // n_heads

        q = x @ self.q_w # q: (batch_size, seq_len, num_heads*head_dim)
        k = x @ self.k_w
        v = x @ self.v_w
        assert q.shape == (batch_size, seq_len, embed_size), f'failed in q shape {q.shape}, and should be {(batch_size, seq_len, embed_size)}'
        #check for nan values
        assert not torch.isnan(q).any(), 'q has nan values'
        assert not torch.isnan(k).any(), 'k has nan values'
        assert not torch.isnan(v).any(), 'v has nan values'

        # attention scores
        q = q.view(batch_size, seq_len, n_heads, head_dim) # q: (batch_size, seq_len, num_heads, head_dim)
        k = k.view(batch_size, seq_len, n_heads, head_dim)
        v = v.view(batch_size, seq_len, n_heads, head_dim)
        assert q.shape == (batch_size, seq_len, n_heads, head_dim), f'failed in q shape {q.shape}, and should be {(batch_size, seq_len, n_heads, head_dim)}'
        
        # b = batch q,k = seq_len, h = n_heads, d = head_dim
        # multiply q (seq_len * head_dim) @ k (head_dim * seq_len) =>  seq_len * seq_len
        scores = torch.einsum('bqhd,bkhd->bhqk', [q, k])
        assert scores.shape == (batch_size, n_heads, seq_len, seq_len), f'failed in scores shape {scores.shape}, and should be {(batch_size, n_heads, seq_len, seq_len)}'
        
        # mask and softmax
        scores = torch.nn.functional.softmax(scores, dim=-1)
        # assert that sums to one, this sometimes fails due to nan values
        # assert round(sum(scores[0][0][0]).item(),2)  == 1, f'failed in sum of scores {sum(scores[0][0][0])}, and should be {1}'
        
        # scores shape: (batch_size, n_heads, seq_len, seq_len)
        # v shape:      (batch_size, seq_len, n_heads, head_dim) 
        # batch_size , n _head are independant
        # multiplication is therefore scores(seq_len, seq_len) @ v(seq_len, head_dim) => (seq_len, head_dim)
        out = torch.einsum('bhqk,bkhd->bqhd', [scores, v])
        assert out.shape == (batch_size, seq_len, n_heads, head_dim), f'failed in out shape {out.shape}, and should be {(batch_size, seq_len, n_heads, head_dim)}'

        # Reshape back to 3 dimensions: (batch_size, seq_len, embed_size)
        out = out.reshape(batch_size, seq_len, -1)
        assert out.shape == (batch_size, seq_len, embed_size), f'failed in out shape {out.shape}, and should be {(batch_size, seq_len, embed_size)}'

        # linear proj, residual, layer norm
        out = self.linear_out(out)
        out = out + x
        out = self.ln(out)

        # feedforward
        out = self.ff(out)
        out = out + x # residual
        out = self.ln2(out)
        assert out.shape == (batch_size, seq_len, embed_size), f'failed in out shape {out.shape}, and should be {(batch_size, seq_len, embed_size)}'
        return out
    

### PARAMATERS ###
params = {
    'epochs': 4,
    'batch_size': 150,
    'num_transformers': 5, # number of transformer layers
    'seq_len': 45,
    'vocab_size': 30522,
    'embed_size': 512 ,
    'n_heads': 32,
    'output_dim': 30522,
    'hidden_size': 1024, # feedforward network hidden size
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'

}

# Load the tinystories dataset
tinystories_dataset = load_dataset("roneneldan/TinyStories")
enc = BertTokenizer.from_pretrained("bert-base-uncased")

device = params['device']
assert params['embed_size'] % params['n_heads'] == 0, f'embed_size {params["embed_size"]} should be divisible by num_heads {params["n_heads"]}'


### OBJECTS ###
train_dataset = SlidingWindowDataset(
    tinystories_dataset["train"]["text"],
    params
)
val_dataset = SlidingWindowDataset(
    tinystories_dataset["validation"]["text"], 
    params
)
train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

model = MyModel(params).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=params['learning_rate'], 
    weight_decay=params['weight_decay']
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    patience=params['patience'], 
    factor=0.1,
    min_lr=1e-11,
    verbose=True
)

def generate_text(model, tokenizer, num_tokens_to_generate: int = 50):
    model.eval()  # Set the model to evaluation mode
    original_batch_size = model.params['batch_size']
    model.params['batch_size'] = 1 # we only want to generate one story (avoids assertion error)
    x = "Once apon a time there was a boy called tim. \
        he had a sister called lilly. they liked to play with their toys. \
        one day they decided to go to the park. \
        they played on the swings and the slide." 

    x = tokenizer.encode(x, add_special_tokens=False, return_tensors='pt')
    expected_seq_len = model.params['seq_len']
    if x.size(1) > expected_seq_len:
        x = x[:, -expected_seq_len:]
    elif x.size(1) < expected_seq_len:
        padding = torch.zeros((1, expected_seq_len - x.size(1)), dtype=torch.long)
        x = torch.cat([padding, x], dim=1)

    initial_text = tokenizer.decode(x.tolist()[0])
    x = x.to(device)
    generated_tokens = []
    with torch.no_grad():  
        for _ in range(num_tokens_to_generate):
            output = model(x)  # Forward pass output:(batch size, vocab size)
            # Convert logits to probabilities
            probs = softmax(output, dim=-1)
            sampled_token = torch.multinomial(probs, num_samples=1).squeeze()
            generated_tokens.append(sampled_token.item())
            new_token = torch.tensor([[sampled_token]], dtype=torch.long).to(device)
            x = torch.cat([x, new_token], dim=1)

            # If the sequence length exceeds the model's maximum, remove the earliest token
            if x.size(1) > model.params['seq_len']:
                x = x[:, 1:]

    model.params['batch_size'] = original_batch_size # reset batch size
    model.train() 
    generated_text = tokenizer.decode(generated_tokens)
    print('----------------------------------')
    print(initial_text)
    print()
    print(generated_text)
    print('----------------------------------')

def evaluate(model, val_dataloader, criterion, n_samples=1, debug=False, generate_text=False):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for i, (x, y ) in enumerate(val_dataloader):
            if x.shape[0] != params['batch_size']:
                print('Error in evaluate data loader, x shape: ', x.shape)
                continue
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            val_loss.append(loss.item())
            if i >= n_samples:
                break 

    loss = sum(val_loss) / len(val_loss)

    if generate_text:
        generate_text(model, val_dataloader, enc)
    
    if debug:
        print(f"Validation loss: {loss}")
        print(f"Input story: {enc.decode(x[0])}")
        print(f"Target: {enc.decode([y[0].tolist()])}")
        print(f"Prediction: {enc.decode([y_pred[0].argmax().tolist()])}")
        print(f"X shape;{x.shape}, y shape;{y.shape}, y_pred shape;{y_pred.shape}")
        print(f"x tensor[0] ()\n{x[0]}")
        print(f"q matrix weights \n{model.q_w}")
    
    model.train()
    return loss
    
loss_history = []
val_loss_history = []



for epoch in range(params['epochs']):  
    for i, (x, y) in enumerate(train_dataloader): # for batch in train_dataloader
        if x.shape[0] != params['batch_size']:
            print(f"Batch {i+1} size: {x.shape}") #Batch 1767 size: torch.Size([289, 50])
            continue

        model.train()
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if i % 3000 == 1:
            generate_text(model, enc)

        if i % 750 == 1:
            percentage = ((epoch + (i / len(train_dataloader))) / params['epochs']) * 100
            recent_loss = sum(loss_history[-100:]) / 100
            val_loss = evaluate(model, val_dataloader, criterion, n_samples=25, debug=False)
            val_loss_history.append(val_loss)
            scheduler.step(val_loss)

            # if lowest loss, save model
            if val_loss == min(val_loss_history):
                torch.save(model.state_dict(), 'model_v2.pth')
                print('Saved best model')
                
            print(f"Epoch {epoch+1}, Iteration {i}, {percentage:.2f}% complete")
            print(f"Train loss: {recent_loss:.5f}")
            print(f"Validation loss: {val_loss:.5f}")
            print()


for i in [1,2,3,4,5]:
    generate_text(model, enc, num_tokens_to_generate=50)
    print()

print('now using best model')
model.load_state_dict(torch.load('model_v2.pth'))

for i in [1,2,3,4,5]:
    generate_text(model, enc, num_tokens_to_generate=50)
    print()

plt.plot(loss_history)
plt.show()
plt.plot(val_loss_history)
plt.show()

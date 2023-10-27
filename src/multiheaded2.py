from typing import List, Tuple, Dict, Set, Union, Any, cast, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
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


# class Head(nn.Module):
#     def __init__(self,seq_len, embedding_size, head_size) -> None:
#         super().__init__()
#         self.seq_len = seq_len
#         self.embedding_size = embedding_size
#         self.head_size = head_size
#         self.k  = nn.Linear(embedding_size, head_size, bias=False)
#         self.q  = nn.Linear(embedding_size, head_size, bias=False)
#         self.v  = nn.Linear(embedding_size, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
#         self.dropout = nn.Dropout()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         k = self.k(x)
#         q = self.q(x)
#         v = self.v(x)
#         scores = q @ k.T
#         scores = scores / math.sqrt(self.head_size)
#         scores = scores.masked_fill(self.tril == 0, float('-inf'))
#         scores = F.softmax(scores, dim=1)
#         scores = self.dropout(scores)
#         out = scores @ v
#         assert out.shape == (self.seq_len, self.head_size)
#         return out

class Head(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, head_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.k = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.q = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.v = nn.Linear(embedding_size, head_size, bias=False).to(device)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)).to(device))
        self.dropout = nn.Dropout().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        # Minibatch dimension is now x.shape[0]
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        scores = torch.einsum('bse,bte->bst', q, k)
        scores = scores / math.sqrt(self.head_size)
        tril = self.tril[None, :, :].expand(scores.size(0), -1, -1).to(device)  # Adding minibatch dimension to tril
        scores = scores.masked_fill(tril == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        out = torch.einsum('bst,bte->bse', scores, v)
        
        # The expected output shape has a minibatch dimension now
        assert out.shape == (x.shape[0], self.seq_len, self.head_size), f"Unexpected output shape {out.shape}"
        return out
    
# class FeedFoward(nn.Module):
#     def __init__(self, embedding_size):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(embedding_size, 4 * embedding_size),
#             nn.ReLU(),
#             nn.Linear(4 * embedding_size, embedding_size),
#             nn.Dropout(),
#         )

#     def forward(self, x):
#         out = self.net(x)
#         return out

class FeedForward(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size).to(device),
            nn.ReLU().to(device),
            nn.Linear(4 * embedding_size, embedding_size).to(device),
            nn.Dropout().to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        out = self.net(x)
        return out
    

# class MultiHead(nn.Module):
#     def __init__(self, embedding_size, n_heads, head_size, seq_len):
#         super().__init__()

#         self.heads = nn.ModuleList([Head(seq_len, embedding_size, head_size) for _ in range(n_heads)])
#         self.proj = nn.Linear(n_heads * head_size, embedding_size,bias=False)
#         self.ln1 = nn.LayerNorm(embedding_size)
#         self.ff = FeedForward(embedding_size)
#         self.ln2 = nn.LayerNorm(embedding_size)
    
#     def forward(self, x):
#         heads = [head(x) for head in self.heads]
#         out = torch.cat(heads, dim=1)
#         out = self.proj(out)
#         out = x + out
#         out = self.ln1(out)
#         out = self.ff(out)
#         out = out + x
#         out = self.ln2(out)
#         return out


class MultiHead(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, head_size: int, seq_len: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_len, embedding_size, head_size).to(device) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, embedding_size, bias=False).to(device)
        self.ln1 = nn.LayerNorm(embedding_size).to(device)
        self.ff = FeedForward(embedding_size).to(device)
        self.ln2 = nn.LayerNorm(embedding_size).to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        heads = [head(x) for head in self.heads]
        out = torch.cat(heads, dim=-1)  # Concatenation along the last dimension (head_size)
        out = self.proj(out)
        out = x + out  # Residual connection
        out = self.ln1(out)
        out = self.ff(out)
        out = out + x  # Residual connection
        out = self.ln2(out)
        return out


# class Transformer(nn.Module):
#     def __init__(self,params):
#         super().__init__()
#         vocab_size = params['vocab_size']
#         embedding_size = params['embedding_size']
#         n_heads = params['n_heads']
#         head_size = params['head_size']
#         n_layers = params['n_layers']
#         seq_len = params['seq_len']
#         self.seq_len = params['seq_len']

#         self.embedding = nn.Embedding(vocab_size, embedding_size)
#         self.position_embedding_table = nn.Embedding(seq_len, embedding_size)
#         self.attn_blocks = [MultiHead(embedding_size, n_heads, head_size,seq_len) for _ in range(n_layers)]
#         self.vocab_proj = nn.Linear(embedding_size, vocab_size)

#     def forward(self, x):
#         embedded = self.embedding(x)
#         pos_emb = self.position_embedding_table(torch.arange(self.seq_len))
#         x = embedded + pos_emb
#         for block in self.attn_blocks:
#             x = block(x)
#         x = self.vocab_proj(x)
#         return x
    # def generate(self):
    #     x = torch.zeros(self.seq_len, dtype=torch.long)
    #     text = ''
    #     for i in range(50):
    #         out = self.forward(x)
    #         out = out[-1]
    #         out = out.softmax(dim=0)
    #         #multinomial sampling
    #         idx = torch.multinomial(out, num_samples=1)
    #         x = torch.cat([x[1:], idx])
    #         c = t_to_c[idx.item()]
    #         text += c
    #     return text
    # def train_model(
    #     self,
    #     p: Dict[str, Any]
    # ):
    #     # train_loader = DataLoader(p['train_dataset'], batch_size=p['batch_size'], shuffle=True)
    #     optimizer = p['optimizer'](self.parameters(), lr=p['lr'])
    #     scheduler = p['scheduler'](optimizer,patience=p['patience'])
    #     loss_fn = p['loss_fn']

    #     accumulated_loss = 0.0
    #     losses = []

    #     # Warmup Phase
    #     if p['warmup_steps'] is not None:
    #         print("Starting warmup")
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = 0.01 * p['lr']
    #         for i, (x, y) in enumerate(p['train_dataset']):
    #             if i >= p['warmup_steps']:
    #                 break
    #             out = self(x)
    #             loss = loss_fn(out, y)
    #             loss.backward()
    #             accumulated_loss += loss.item()
    #             if (i + 1) % p['gradient_accumulation_steps'] == 0:
    #                 optimizer.step()
    #                 accumulated_loss = 0.0
    #                 optimizer.zero_grad()
    #         print("Finished warmup")
    #         for param_group in optimizer.param_groups:
    #             param_group['lr'] = p['lr']
            
    #     accumulated_loss = 0.0
    #     for epoch in range(p['epochs']):
    #         for i, (x, y) in enumerate(p['train_dataset']):

    #             out = self(x)
    #             loss = loss_fn(out, y)
    #             loss.backward()
    #             accumulated_loss += loss.item()

    #             # Ensuring that batch_size and gradient_accumulation_steps line up
    #             if (i + 1) % p['gradient_accumulation_steps'] == 0:
    #                 optimizer.step()
    #                 avg_loss = accumulated_loss / p['gradient_accumulation_steps']
    #                 losses.append(avg_loss)
    #                 accumulated_loss = 0.0
    #                 optimizer.zero_grad()
    #                 scheduler.step(avg_loss)

    #             if (i + 1) % 1000 == 0:
    #                 print("Average loss:", avg_loss)
    #                 print(f"percent complete: {(i+1)/len(p['train_dataset']) * 100:.2f}%")
    #                 print(self.generate())
    #             if i > 2000:
    #                 break
    #         print(f"Epoch {epoch} complete")
    #     return losses
    
    # def train_model_profile(self, p: Dict[str, Any]):
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         with record_function("model_training"):
    #             losses = self.train_model(p)
    #     return losses, prof
    

    

class Transformer(nn.Module):
    def __init__(self, params: dict) -> None:
        super().__init__()
        vocab_size = params['vocab_size']
        embedding_size = params['embedding_size']
        n_heads = params['n_heads']
        head_size = params['head_size']
        n_layers = params['n_layers']
        seq_len = params['seq_len']
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embedding_size).to(device)
        self.position_embedding_table = nn.Embedding(seq_len, embedding_size).to(device)
        self.attn_blocks = nn.ModuleList([MultiHead(embedding_size, n_heads, head_size, seq_len).to(device) for _ in range(n_layers)])
        self.vocab_proj = nn.Linear(embedding_size, vocab_size).to(device)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized Transformer with {n_params} parameters on device {device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device)
        batch_size = x.shape[0]
        
        embedded = self.embedding(x)
        pos_emb = self.position_embedding_table(torch.arange(self.seq_len).to(device))
        pos_emb = pos_emb[None, :, :].expand(batch_size, -1, -1)  # Adding minibatch dimension
        x = embedded + pos_emb
        
        for block in self.attn_blocks:
            x = block(x)
        
        x = self.vocab_proj(x)
        return x


    def generate(self) -> str:
        x = torch.zeros(self.seq_len, dtype=torch.long).to(device)
        text = ''
        for _ in range(50):
            out = self.forward(x.unsqueeze(0))  # Add minibatch dimension
            out = out.squeeze(0)[-1]  # Remove minibatch dimension and select the last element
            out = out.softmax(dim=0)
            idx = torch.multinomial(out, num_samples=1)
            x = torch.cat([x[1:], idx])
            c = t_to_c[idx.item()]  # Assuming t_to_c is a token-to-character mapping
            text += c
        return text

    def train_model(self, p: Dict[str, Any]):
        optimizer = p['optimizer'](self.parameters(), lr=p['lr'])
        scheduler = p['scheduler'](optimizer, patience=p['patience'])
        loss_fn = p['loss_fn']
        accumulated_loss = 0.0
        losses = []
        batch_size = p['batch_size']
        train_loader = DataLoader(p['train_dataset'], batch_size=batch_size, shuffle=True)
        

        # Warmup Phase
        if p['warmup_steps'] is not None:
            print("Starting warmup")
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.01 * p['lr']
            for i, (x, y) in enumerate(train_loader):
                if i >= p['warmup_steps']:
                    break
                x, y = x.to(device), y.to(device)
                out = self(x)
                loss = loss_fn(out.view(-1, out.size(2)), y.view(-1))

                # loss = loss_fn(out, y)
                loss.backward()
                accumulated_loss += loss.item()
                if (i + 1) % p['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
            print("Finished warmup")
            for param_group in optimizer.param_groups:
                param_group['lr'] = p['lr']
        
        # Main training loop
        best_loss = float('inf')
        accumulated_loss = 0.0
        for epoch in range(p['epochs']):
            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(device), y.to(device)
                out = self(x)
                loss = loss_fn(out.view(-1, out.size(2)), y.view(-1))
                # loss = loss_fn(out, y)
                loss.backward()
                accumulated_loss += loss.item()

                if (i + 1) % p['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    avg_loss = accumulated_loss / p['gradient_accumulation_steps']
                    losses.append(avg_loss)
                    accumulated_loss = 0.0
                    optimizer.zero_grad()
                    scheduler.step(avg_loss)


                if (i + 1) % 50 == 0:
                    percent_complete = ((epoch +1) / p['epochs']) * ((i+1)/len(p['train_dataset'])) * 100
                    print(f"Epoch: {epoch+1}, iter {i+1} of {len(p['train_dataset'])/ batch_size}")
                    print("Average loss:", avg_loss)
                    print(f"percent complete: {percent_complete:.2f}%")
                    print(self.generate())
                    print()

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        print("Saving model")
                        torch.save(self.state_dict(), 'model_v2.pt')
            print("\n-----------------------------------\n")

        return losses

    def train_model_profile(self, p: Dict[str, Any]):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_training"):
                losses = self.train_model(p)
        return losses, prof
    

class SlidingWindowDataset(Dataset):
    def __init__(self, stories: list, seq_len: int):
        super(SlidingWindowDataset, self).__init__()
        self.seq_len = seq_len
        self.stories = ' \n'.join([s for s in stories[:400_000]])
        self.stories =self.stories.lower()
        self.tokenized_data = [c_to_t[c] for c in self.stories if c in characters]
        print(f"Dataset has {len(self.tokenized_data)} tokens")

    def __len__(self) -> int:
        # Adjust the length so that the last data point has at least seq_len elements remaining
        return len(self.tokenized_data) - 2 * self.seq_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate the ending index for x and y
        end_idx_x = idx + self.seq_len
        end_idx_y = idx + 1 + self.seq_len

        # Get the actual data
        x_data = self.tokenized_data[idx: end_idx_x]
        y_data = self.tokenized_data[idx + 1: end_idx_y]
        
        # Pad if necessary
        x_data += [0] * (self.seq_len - len(x_data))
        y_data += [0] * (self.seq_len - len(y_data))
        
        # Convert to tensors
        x = torch.tensor(x_data, dtype=torch.long)
        y = torch.tensor(y_data, dtype=torch.long)
        
        assert len(x) == len(y) == self.seq_len
        return x, y
    

characters = list(' abcdefghijklmnopqrstuvwxyz.,?!\'"0123456789')
c_to_t = {c: i for i, c in enumerate(characters)}
t_to_c = {i: c for i, c in enumerate(characters)}
tinystories_dataset = load_dataset("roneneldan/TinyStories")
tinystories_dataset = tinystories_dataset['train']['text']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_params = {
    'vocab_size': len(characters),
    'seq_len': 128,
    'embedding_size': 512,
    'n_heads': 8,
    'head_size': 64,
    'n_layers': 6, # transformer blocks
}
assert model_params['embedding_size'] / model_params['n_heads'] == model_params['head_size']

model = Transformer(model_params)

train_params = {
    'epochs': 2,
    'gradient_accumulation_steps': 8,
    'warmup_steps': 2000,
    'lr': 0.0001,
    'batch_size': 100,
    'train_dataset': SlidingWindowDataset(tinystories_dataset, model_params['seq_len']),
    'optimizer': torch.optim.Adam,
    'loss_fn': nn.CrossEntropyLoss(),
    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'patience': 5,
}


out = model.generate()
print("\n", out, "\n")
losses = model.train_model(train_params)

plt.plot(losses)
plt.show()
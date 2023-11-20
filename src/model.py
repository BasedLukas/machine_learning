from typing import List, Tuple, Dict, Set, Union, Any, cast, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
from collections import Counter
import os
import pickle
import sys
from data import SlidingWindowDataset

# x: (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, head_size)
class Head(nn.Module):
    def __init__(self, seq_len: int, embedding_size: int, head_size: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.embedding_size = embedding_size
        self.head_size = head_size
        self.k = nn.Linear(embedding_size, head_size, bias=False)
        self.q = nn.Linear(embedding_size, head_size, bias=False)
        self.v = nn.Linear(embedding_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_len, seq_len)))
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        scores = torch.bmm(q, k.transpose(1, 2))
        scores = scores / math.sqrt(self.head_size)
        tril = self.tril[None, :, :].expand(scores.size(0), -1, -1)
        scores = scores.masked_fill(tril == 0, float('-inf'))
        scores = F.softmax(scores, dim=-1)
        out = torch.bmm(scores, v)
        out = self.dropout(out)
        return out
    
# X: (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, embedding_size)
class FeedForward(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU().to(device),
            nn.Linear(4 * embedding_size, embedding_size),
            nn.Dropout().to(device)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out
    
# X: (batch_size, seq_len, embedding_size) -> (batch_size, seq_len, embedding_size)
class MultiHead(nn.Module):
    def __init__(self, embedding_size: int, n_heads: int, head_size: int, seq_len: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(seq_len, embedding_size, head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, embedding_size, bias=False)
        self.ln1 = nn.LayerNorm(embedding_size)
        self.ff = FeedForward(embedding_size)
        self.ln2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        heads = [head(x) for head in self.heads]
        out = torch.cat(heads, dim=-1)  # Concatenation along the last dimension (head_size)
        out = self.proj(out)
        out = x + out  # Residual connection
        out = self.ln1(out)
        out = self.ff(out)
        out = out + x  # Residual connection
        out = self.ln2(out)
        return out


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

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding_table = nn.Embedding(seq_len, embedding_size).to(device)
        self.attn_blocks = nn.ModuleList([MultiHead(embedding_size, n_heads, head_size, seq_len) for _ in range(n_layers)])
        self.vocab_proj = nn.Linear(embedding_size, vocab_size)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized Transformer with {n_params:,} parameters on device {device}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        embedded = self.embedding(x)
        # pos_emb = self.position_embedding_table(torch.arange(self.seq_len))
        pos_emb = self.position_embedding_table(torch.arange(self.seq_len).to(device))

        pos_emb = pos_emb[None, :, :].expand(batch_size, -1, -1) 
        x = embedded + pos_emb
        
        for block in self.attn_blocks:
            x = block(x)
        
        x = self.vocab_proj(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    model_params = {
        'batch_size': 2,
        'vocab_size': 30522,
        'seq_len': 128,
        'embedding_size': 512,
        'n_heads': 8,
        'head_size': 64,
        'n_layers': 6, # transformer blocks
        'tokenized_data':'tokenized_data_train.pkl',
        'max_dataset_tokens': 100_000_000
    }
    assert model_params['embedding_size'] / model_params['n_heads'] == model_params['head_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(model_params).to(device)
    dataset = SlidingWindowDataset([],model_params)
    dataloader = DataLoader(dataset, batch_size=model_params['batch_size'], shuffle=True, num_workers=0)
    del dataset

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        print('starting forward pass')
        out = model(x)
        print('finished forward pass')
        break
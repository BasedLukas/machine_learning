
from typing import List
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
import math

from data_prep import Tokenizer, Dataset



class AttentionHead(nn.Module):
    def __init__(self, context_window, embedding_dim):
        super(AttentionHead, self).__init__()
        self.context_window, self.embedding_dim = context_window, embedding_dim
        
        # Attention
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=0)
        self.norm = nn.LayerNorm(embedding_dim)

        
    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window, self.embedding_dim), f"input shape is {x.shape} and should be {(self.context_window, self.embedding_dim)}"

        #attention
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0, 1)) / (self.embedding_dim ** 0.5)
        attention = self.softmax(scores)
        weighted = torch.mm(attention, values)

        weighted = weighted + queries
        weighted = self.norm(weighted)
        assert weighted.shape == (self.context_window, self.embedding_dim), f"weighted shape is {weighted.shape} and should be {(self.context_window,self.embedding_dim)}"
        return weighted




class EmbeddingModel(nn.Module):
    def __init__(self, context_window, embedding_dim, vocab_size):
        super(EmbeddingModel, self).__init__()
        self.context_window, self.embedding_dim, self.vocab_size = context_window, embedding_dim, vocab_size
        positional_encodings = self._generate_positional_encodings()
        self.register_buffer('positional_encodings', positional_encodings)
        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    

    
    def forward(self, x:torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window,), f"input shape is {x.shape} and should be {self.context_window,}"

        #embeddings
        x = self.embeddings(x)
        x = x + self.positional_encodings
        assert x.shape == (self.context_window, self.embedding_dim), f"embedding shape is {x.shape} and should be {(self.context_window,self.embedding_dim)}"
        return x
    

    
    def _generate_positional_encodings(self):
        # Initialize positional encoding matrix
        pos_encodings = torch.zeros(self.context_window, self.embedding_dim)
        # Generate encoding for each position
        for pos in range(self.context_window):
            for i in range(0, self.embedding_dim, 2):
                pos_encodings[pos, i] = math.sin(pos / 10000 ** (i / self.embedding_dim))
                if i + 1 < self.embedding_dim:  # Ensure index is in range
                    pos_encodings[pos, i + 1] = math.cos(pos / 10000 ** ((i + 1) / self.embedding_dim))
        return pos_encodings


class MLP(nn.Module):
    def __init__(self, context_window, embedding_dim, hidden_dim, vocab_size,dropout=0.1):
        super(MLP, self).__init__()
        self.context_window, self.embedding_dim, self.hidden_dim, self.vocab_size = context_window, embedding_dim, hidden_dim, vocab_size
        
        # Feedforward MLP with non-linearity
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  
            nn.LeakyReLU(),  
            nn.Dropout(dropout),                           
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window, self.embedding_dim), f"input shape is {x.shape} and should be {(self.context_window, self.embedding_dim)}"

        #feedforward
        output = self.feedforward(x)


        assert output.shape == (self.context_window, self.embedding_dim), f"output shape is {output.shape} and should be {(self.context_window, self.embedding_dim)}"
        
        return output


class TransformerModel(nn.Module):
    def __init__(self, context_window, embedding_dim, hidden_dim, vocab_size,dropout=0.1):
        super(TransformerModel, self).__init__()
        self.context_window, self.embedding_dim, self.hidden_dim, self.vocab_size = context_window, embedding_dim, hidden_dim, vocab_size
        
        self.embeddings = EmbeddingModel(context_window, embedding_dim, vocab_size)
        self.attention = AttentionHead(context_window, embedding_dim)
        self.feedforward = MLP(context_window, embedding_dim, hidden_dim, vocab_size,dropout=dropout)

    def forward(self, x:torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window,), f"input shape is {x.shape} and should be {self.context_window,}"

        #embeddings
        x = self.embeddings(x)
        assert x.shape == (self.context_window, self.embedding_dim), f"embedding shape is {x.shape} and should be {(self.context_window,self.embedding_dim)}"

        #attention
        x = self.attention(x)
        assert x.shape == (self.context_window, self.embedding_dim), f"attention shape is {x.shape} and should be {(self.context_window,self.embedding_dim)}"

        #feedforward
        x = self.feedforward(x)
        # assert x.shape == (self.vocab_size,), f"output shape is {x.shape} and should be {self.vocab_size}"
        
        return x
    


class MultiLayerTransformer(nn.Module):
    def __init__(self, context_window, embedding_dim, hidden_dim, vocab_size, num_layers=3, dropout=0.1):
        super(MultiLayerTransformer, self).__init__()
        self.context_window, self.embedding_dim, self.hidden_dim, self.vocab_size = context_window, embedding_dim, hidden_dim, vocab_size

        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                EmbeddingModel(context_window, embedding_dim, vocab_size),
                AttentionHead(context_window, embedding_dim),
                MLP(context_window, embedding_dim, hidden_dim, vocab_size, dropout=dropout),
                nn.Linear(embedding_dim, vocab_size),
            ) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window,), f"input shape is {x.shape} and should be {self.context_window,}"

        # Apply each feedforward layer independently
        outputs = [layer(x) for layer in self.feedforward_layers]
        
        # Sum outputs
        x = torch.stack(outputs).sum(dim=0)
        x = x.squeeze(0)
        x = x.mean(dim=0)


        assert x.shape == (self.vocab_size,), f"output shape is {x.shape} and should be {self.vocab_size}"

        return x



    def prepare_input(self, x: List[int])-> torch.Tensor:
        padding_token_id = 0

        x = x[-self.context_window:]
        len_padding = max(0, self.context_window - len(x))
        padding = [padding_token_id] * len_padding
        x = padding + x
        assert len(x) == self.context_window, f"input length is {len(x)} and should be {self.context_window}"
        x = torch.tensor(x)
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, embedding_dim: int, context_window):
        super().__init__()


        position = torch.arange(context_window).unsqueeze(1)
        print('position', position)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim))
        print('div_term', div_term)
        pe = torch.zeros(context_window,  embedding_dim)
        print('pe', pe)
        pe = torch.sin(position * div_term)
        print('pe[0, 0::2]', pe)
        pe = pe + torch.cos(position * div_term)
        print('pe[0, 1::2]', pe)
        self.register_buffer('pe', pe)
        print(pe.shape,'pe shape')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe
        return x

token = Tokenizer()
dataset = Dataset(token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


context_window = 4
vocab_size = token.vocab_size
embedding_dim = 8
hidden_dim = 512
lr = 0.001
dropout = 0.01
max_norm = 0.5
num_layers = 6

    
pe = PositionalEncoding(embedding_dim=embedding_dim,context_window=context_window)
x, y = dataset.get_training_data()

model = MultiLayerTransformer(context_window=context_window, 
                            embedding_dim=embedding_dim, 
                            hidden_dim=hidden_dim, 
                            vocab_size=vocab_size,
                            num_layers=num_layers,
                            dropout=dropout
                            ).to(device)
x = model.prepare_input(x)
print(x.shape,'x shape')
sys.exit()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.5, min_lr=1e-8)

def sample(max_tokens=15):
    x = [185] #sos token
    eos_token = token.encode('<eos>')[0]
    model.eval()
    while x[-1] != eos_token and len(x) < max_tokens:
        with torch.no_grad():
            x_tensor = model.prepare_input(x).to(device)
            output = model(x_tensor)
            output = F.softmax(output, dim=0)
            output = torch.multinomial(output, num_samples=1)
            output = output.item()
            x.append(output)
    
    model.train()
    x = token.decode(x)
    print(x)

    return x

def train_loop():
    print('running on device: ', device)
    print(f'model params {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print()
    loss_tracker = []
    for i in range(100_000):
        model.train()
        x, y = dataset.get_training_data()
        x = model.prepare_input(x).to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.unsqueeze(0), torch.tensor(y).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        loss_tracker.append(loss.item())
        if i % 200 == 0:
            mean_loss = np.mean(loss_tracker)
            lr_scheduler.step(mean_loss)
            print(f"iteration: {i}")
            print(f"loss: {mean_loss}")
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
            sample(20)
            
            print()

            loss_tracker = []
    


train_loop()
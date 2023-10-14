
from typing import List, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
import math

from data_prep import Tokenizer, Dataset


# IN (context window)
# OUT (context_window, embedding_dim)
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

# IN (context_window, embedding_dim)
# OUT (context_window, embedding_dim)
class AttentionHead(nn.Module):
    def __init__(self, context_window, embedding_dim):
        super(AttentionHead, self).__init__()
        self.context_window, self.embedding_dim = context_window, embedding_dim
        
        # Attention
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.LayerNorm(embedding_dim)

        
    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window, self.embedding_dim), f"input shape is {x.shape} and should be {(self.context_window, self.embedding_dim)}"

        #attention
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0, 1)) / (self.embedding_dim ** 0.5)
        scores += (mask * -1e9)
        attention = self.softmax(scores)
        weighted = torch.mm(attention, values)

        # check that rows in attention matrix sum to 1
        assert torch.all(torch.sum(attention, dim=1) - torch.ones(self.context_window).to(device) < 1e-5), f"attention matrix rows do not sum to 1"

        weighted = weighted + queries
        weighted = self.norm(weighted)
        assert weighted.shape == (self.context_window, self.embedding_dim), f"weighted shape is {weighted.shape} and should be {(self.context_window,self.embedding_dim)}"
        return weighted


# IN (context_window, embedding_dim)
# OUT (context_window, embedding_dim)
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
    
# IN List of ints
# OUT (vocab_size,)
class Transformer(nn.Module):
    def __init__(self,context_window, embedding_dim, hidden_dim, vocab_size, num_layers, dropout=0.1):
        super(Transformer, self).__init__()
        self.context_window, self.embedding_dim, self.hidden_dim, self.vocab_size, self.num_layers = context_window, embedding_dim, hidden_dim, vocab_size, num_layers
        self.attentions = nn.ModuleList([AttentionHead(context_window, embedding_dim) for _ in range(num_layers)])
        self.mlps = nn.ModuleList([MLP(context_window, embedding_dim, hidden_dim, vocab_size, dropout=dropout) for _ in range(num_layers)])
        
        self.embedding = EmbeddingModel(context_window, embedding_dim, vocab_size)
        self.output_layer = nn.Linear(self.context_window * self.embedding_dim, self.vocab_size)


    def prepare_input(self, x: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        padding_token_id = 0
        x = x[-self.context_window:]
        len_padding = max(0, self.context_window - len(x))
        padding = [padding_token_id] * len_padding
        x = padding + x

        # Create a mask with '1's where there is padding and '0' otherwise
        mask = [1 if token == padding_token_id else 0 for token in x]

        assert len(x) == self.context_window, f"input length is {len(x)} and should be {self.context_window}"
        x = torch.tensor(x).to(device)
        mask = torch.tensor(mask).to(device)

        return x, mask
    
    
    def forward(self, x: torch.Tensor):
        x, mask = self.prepare_input(x)
        assert isinstance(x, torch.Tensor), f"input type is {type(x)} and should be a torch.Tensor"
        assert x.shape == (self.context_window,), f"input shape is {x.shape} and should be {(self.context_window,)}"
        
        #embedding
        x = self.embedding(x)
        assert x.shape == (self.context_window, self.embedding_dim), f"embedding shape is {x.shape} and should be {(self.context_window, self.embedding_dim)}"
        #attention
        for i in range(self.num_layers):
            x = self.attentions[i](x, mask)
            x = self.mlps[i](x)
        assert x.shape == (self.context_window, self.embedding_dim), f"output shape is {x.shape} and should be {(self.context_window, self.embedding_dim)}"

        # final output layer to convert to 1 d tensor of vocab size
        x = x.view(-1)  # Flatten the tensor
        x = self.output_layer(x)
        assert x.shape == (self.vocab_size,), f"output shape is {x.shape} and should be {(self.vocab_size,)}"

        return x


def sample(model, token, max_tokens=15):
    """Used to sample from the model. Prints the output."""
    x = token.encode('<sos>')
    eos_token = token.encode('<eos>')[0]
    model.eval()
    while x[-1] != eos_token and len(x) < max_tokens:
        with torch.no_grad():
            
            output = model(x)
            output = F.softmax(output, dim=0)
            output = torch.multinomial(output, num_samples=1)
            output = output.item()
            x.append(output)
    
    model.train()
    x = token.decode(x)
    print(x)
    return x



if __name__ == '__main__':
    token = Tokenizer()
    dataset = Dataset(token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### PARAMS ###
    context_window = 8
    vocab_size = token.vocab_size
    embedding_dim = 512
    hidden_dim = 512
    lr = 0.001
    dropout = 0.0001
    max_norm = 0.8
    num_layers = 16
    iterations = 70_000
    patience = 700
    weight_decay = 0.0001
    momentum = 0.7

    model = Transformer(
        context_window=context_window, 
        embedding_dim=embedding_dim, 
        hidden_dim=hidden_dim, 
        vocab_size=vocab_size,
        num_layers=num_layers,
        dropout=dropout
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=patience, 
        verbose=True, 
        factor=0.5, 
        min_lr=1e-10
    )


    print('running on device: ', device)
    print(f'model params {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print()
    history = []
    loss_total = 0
    for i in range(iterations):
        model.train()
        x, y = dataset.get_training_data()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.unsqueeze(0), torch.tensor(y).to(device))
        if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
            print("NaN or Inf in loss, skipping update.")
            continue
        loss.backward()

        # Check gradient magnitudes
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         grad_norm = param.grad.norm()
        #         if grad_norm.item() == 0:
        #             print(f"Zero gradient at {name}")
        #         if torch.isnan(grad_norm) or torch.isinf(grad_norm):
        #             print(f"NaN or Inf gradient at {name}")
        
        lr_scheduler.step(loss)

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss_total += loss.item()
        if i % 300 == 0 and i != 0:
            print(f"iteration: {i}")
            print(f"loss: {loss_total / 200}")
            history.append(loss_total / 200)
            loss_total = 0
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")
            sample(model, token, 50)
            print()

#plot
import matplotlib.pyplot as plt
plt.plot(history)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
            
    



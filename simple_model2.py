import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple, Dict, Union, Callable, Iterable
from data_prep import Tokenizer, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Configs
        self.embedding_dim = config['embedding_dim']
        self.context_window = config['context_window']
        self.lr = config['lr']
        self.gamma = config['gamma']

        self.t = Tokenizer()
        self.dataset = Dataset(self.t)
        vocab_size = self.t.vocab_size
        self.padding_token_id = self.t.encode('<pad>')[0]

        # Print configs
        print(f'Configurations: {config}')

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        # Linear layers with activation and dropout
        self.layer1 = torch.nn.Linear(self.embedding_dim, 256)
        self.activation = torch.nn.LeakyReLU()
        self.dropout = torch.nn.Dropout(0.2)
        self.layer2 = torch.nn.Linear(256, vocab_size)

        # Optimization and scheduling
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

        # Print architecture details
        print(f'Model initialized with vocabulary size {vocab_size} and embedding dimension {self.embedding_dim}')

        self.to(device)

    def forward(self, x: List[int])->torch.Tensor:
        # Use a separate function for data preparation
        x = prepare_data(x, self.context_window, self.padding_token_id)
        x = torch.tensor(x).to(device)

        # Embedding
        embedded_input = self.embedding(x)

        # Average all the tokens to pool information
        sum_embeddings = torch.sum(embedded_input[x != self.padding_token_id], dim=0)
        num_non_padding_tokens = torch.sum(x != self.padding_token_id)
        average_embedding = sum_embeddings / (num_non_padding_tokens + 1e-10)

        # Pass through layers
        x = self.layer1(average_embedding)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.layer2(x)

        return logits

    def forward_verbose(self, x: List[int])->torch.Tensor:
        x = prepare_data(x, self.context_window, self.padding_token_id)
        x = torch.tensor(x).to(device)
        print('final x is', x)

        # Embedding
        embedded_input = self.embedding(x)
        print('embedded_input shape is', embedded_input.shape)

        # Average all the tokens to pool information
        sum_embeddings = torch.sum(embedded_input[x != self.padding_token_id], dim=0)
        num_non_padding_tokens = torch.sum(x != self.padding_token_id)
        average_embedding = sum_embeddings / (num_non_padding_tokens + 1e-10)

        # Pass through layers
        x = self.layer1(average_embedding)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.layer2(x)

        return logits

    def train_model(self, x, y, verbose=False):
        x = torch.tensor(prepare_data(x, self.context_window, self.padding_token_id)).to(device)
        y = torch.tensor(y).to(device)

        logits = self.forward(x)

        # Compute the loss and update weights
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), y)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()

    def sample(self, max_length=50) -> str:
        # Set the model to evaluation mode
        self.eval()

        # Disable gradient computation
        with torch.no_grad():
            # Initialize the sequence with start token
            start_token = '<sos>'
            output = ''
            x = self.t.encode(start_token)

            # Generate tokens up to max_length
            for _ in range(max_length):
                logits = self.forward(x)
                p = torch.nn.functional.softmax(logits, dim=0)
                sampled_index = torch.multinomial(p, 1).item()

                output += " " + self.t.decode([sampled_index])
                if sampled_index == self.t.encode('<eos>')[0]:
                    break

                x = x[1:] + [sampled_index]

        # Set the model back to training mode
        self.train()

        return output


# Move data preparation outside of the model
def prepare_data(x, context_window, padding_token_id):
    x = x[-context_window:]
    len_padding = max(0, context_window - len(x))
    padding = [padding_token_id] * len_padding
    x = padding + x
    return x



if __name__ == '__main__':


    CONFIG = {  
        'context_window': 16,
        'embedding_dim': 126,
        'lr': 0.5,
        'gamma': 0.9,
    }

    CONFIG['context_window'] = 8

    model = SimpleModel(CONFIG)

    #demonstrate procedure
    long = 'hello this is a ltignmsetloigjnm test of more than 8 words in a sentence'
    encoded = model.t.encode(long)
    print('text:', long)
    print('encoded:', encoded)
    print('decoded:', model.t.decode(encoded))
    print('forward pass')
    model.forward_verbose(encoded)
    print()

    short = 'hello this is a test'
    encoded = model.t.encode(short)
    print('text:', short)
    print('encoded:', encoded)
    print('decoded:', model.t.decode(encoded))
    print('forward pass')
    model.forward_verbose(encoded)   
    print()



    #train model
    CONFIG['context_window'] = 16
    model = SimpleModel(CONFIG)

    for i in range(10_000):

        if i % 100 == 0:
            print('iteration', i)
            model.train(verbose=True)
            model.scheduler.step()
            print()
        else:
            model.train(verbose=False)
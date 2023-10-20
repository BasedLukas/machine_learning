import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import re
from typing import List, Tuple, Dict, Union, Callable, Iterable, Optional
from datasets import load_dataset

### OBJECTS ###
class Tokenizer:
    """Tokenizes on a word level. lower cases everything and removes most special characters. Uses tinystories dataset."""
    def __init__(self, vocab_path='vocab.txt'):
        """
        Initialize the tokenizer with the vocabulary.
        :param vocab_path: path to the vocabulary file
        """
        if not os.path.exists(vocab_path):
            self._create_vocab(vocab_path)
        self.vocab = sorted(list(open(vocab_path, 'r').read().split('\n')))
        # ensure the padding token comes first so that it has index 0
        self.vocab = ['<pad>'] + [word for word in self.vocab if word != '<pad>']
        self.word2index = {word: i for i, word in enumerate(self.vocab)}  # type: Dict[str, int]
        self.index2word = {i: word for i, word in enumerate(self.vocab)}  # type: Dict[int, str]
        self.vocab_size = len(self.vocab)  # type: int

    def _create_vocab(self, vocab_path: str) -> None:
        """
        Create vocabulary from dataset and save it to the file.
        :param vocab_path: path to the vocabulary file
        """
        print('Vocab not found. Creating vocab...')
        dataset = load_dataset("roneneldan/TinyStories")
        vocab = set()
        for i in range(1_000_000):
            text = dataset['train'][i]['text'].lower()
            tokens = re.findall(r'\b[a-zA-Z0-9]+\b|[.,!?;]', text)  # type: List[str]
            vocab.update(tokens)
            
        vocab = (word.strip() for word in vocab if word.isalnum() or word in '.,!?;')
        vocab = ['<unk>', '<sos>', '<eos>', '<pad>'] + sorted(list(vocab))
        with open(vocab_path, 'w') as f:
            f.write('\n'.join(vocab))
        print('vocab created')

    def encode(self, s: str) -> list:
        """
        Encode the string into token indices.
        :param s: input string
        :return: list of token indices
        """

        tokens = re.findall(r'<pad>|<sos>|<eos>|\b\w+\b|[.,!?;]', s.lower())
        # tokens = re.findall(r'<pad>|<sos>|<eos>|\b[a-zA-Z0-9]+\b|[.,!?;]', s.lower())  # type: List[str]
        return [self.word2index[token] if token in self.vocab else self.word2index['<unk>'] for token in tokens]  # type: List[int]

    def decode(self, l: list) -> str:
        """
        Decode the list of token indices into a string.
        :param l: list of token indices
        :return: decoded string
        """
        return " ".join([self.index2word[i] for i in l])  # type: str


class Dataset:
    """Tiny stories dataset. doesn't do batching. Randomly draws from dataset. Ads sos and eos tokens."""
    def __init__(self, tokenizer: Tokenizer):
        """
        Initialize the dataset with a given tokenizer.
        :param tokenizer: Tokenizer object for encoding text
        """
        self.dataset = load_dataset("roneneldan/TinyStories")  # type: dict
        self.tokenizer = tokenizer  # type: Tokenizer
        self.length = len(self.dataset['train'])  # type: int

        self.current_story = None  # type: Optional[str]
        self.current_position = 0   # type: int

    def get_training_data(self) -> Tuple[List[int], List[Optional[int]]]:
        """
        Get training data, including the current story segment (x) and the next token (y).
        :return: Tuple of current segment (x) and next token (y)
        """
        # If no current story is being processed, randomly select one
        if self.current_story is None:
            story_idx = np.random.randint(0, self.length)
            self.current_story = self.dataset['train'][story_idx]['text']  # type: str
            self.current_position = 0  # type: int

        # Add start and end tokens to the story and encode
        story = "<sos> " + self.current_story + " <eos>"
        story = self.tokenizer.encode(story)  # type: List[int]

        # Determine x and y based on the current position
        x = story[0:self.current_position + 1]  # type: List[int]
        y = story[self.current_position + 1] if self.current_position + 1 < len(story) else None  # type: Optional[int]

        # If we've reached the end of the story, reset the current story and position
        if y is None:
            self.current_story = None
            self.current_position = 0
            return self.get_training_data()

        # Increment the current position for the next call
        self.current_position += 1

        return x, [y]  # type: Tuple[List[int], List[int]


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
        self.loss_tracker = []

        # Print configs
        print(f'Configurations: {config}')

        # Embedding layer
        self.embedding_tensor = nn.Parameter(torch.rand(vocab_size, self.embedding_dim) * 0.1)
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
        """takes in tokenized text and returns logits"""

        # Preparing the input with context window and padding
        x = x[-self.context_window:]
        len_padding = max(0, self.context_window - len(x))
        padding = [self.padding_token_id] * len_padding
        x = padding + x
        x = torch.tensor(x).to(device)
        
        # Embedding
        embedded_input = self.embedding_tensor[x]

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
        """Identical with print statements for clarity"""

        print('Input x is', x)
        # Preparing the input with context window and padding
        x = x[-self.context_window:]
        print('After context window, x is', x)
        len_padding = max(0, self.context_window - len(x))
        padding = [self.padding_token_id] * len_padding
        print('Padding is', padding)
        print('Length of the padding is', len_padding),'and length of x is', len(x)
        x = padding + x
        x = torch.tensor(x).to(device)
        print('final x is', x)
        
        # Embedding
        embedded_input = self.embedding_tensor[x]
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


    def train(self, verbose=False):
        # Fetch training data
        x, y = self.dataset.get_training_data()
        y = torch.tensor(y).to(device)


        logits = self.forward(x)

        # Compute the loss and update weights
        loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), y)
        self.loss_tracker.append(loss.item())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Verbose logging
        if verbose:
            mean_loss = np.mean(self.loss_tracker)
            self.loss_tracker = []
            sampled_text = self.sample(15)

            print('Using device:', device)
            print('Mean loss:', mean_loss)
            print('Learning rate:', self.scheduler.get_last_lr()[0])
            print('Sampled text:', sampled_text)
            print()


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


### RUN ###
if __name__ == '__main__':

    CONFIG = {  
        'context_window': 8,
        'embedding_dim': 126,
        'lr': 0.5,
        'gamma': 0.9,
        'epochs': 10_000
    }

    device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = Tokenizer()
    dataset = Dataset(t)
    model = SimpleModel(CONFIG)
    
    print("Demonstrating tokenization and data feed:\n")
    for i in range(5):
        x, y = dataset.get_training_data()
        yd = t.decode(y)
        xd = t.decode(x)
        print('x is', x)
        print('y is', y)
        print('decoded:')
        print('x is: ', xd)
        print('y is: ', yd)
        print()
    print("\n-------------------\n")
    


    #demonstrate procedure
    print('Demonstrating truncation and padding\n')
    long = 'hello this is a test of more than 8 words in a sentence'
    encoded = model.t.encode(long)
    print('text:', long)
    print('encoded:', encoded)
    print('decoded:', model.t.decode(encoded))
    model.forward_verbose(encoded)   
    print()

    short = 'hello this is a test'
    encoded = model.t.encode(short)
    print('text:', short)
    print('encoded:', encoded)
    print('decoded:', model.t.decode(encoded))
    model.forward_verbose(encoded)   
    print()



    #train model with a longer context window
    print('Re-init model with new params and train:')
    CONFIG['context_window'] = 16
    model = SimpleModel(CONFIG)

    for i in range(CONFIG['epochs']):

        if i % 100 == 0:
            print('iteration', i)
            model.train(verbose=True)
            model.scheduler.step()
            print()
        else:
            model.train(verbose=False)
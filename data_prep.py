import numpy as np
import os
from datasets import load_dataset
import re

from typing import Dict, List, Tuple, Union, Optional



class Tokenizer:
    def __init__(self, vocab_path='vocab.txt'):
        """
        Initialize the tokenizer with the vocabulary.
        :param vocab_path: path to the vocabulary file
        """
        if not os.path.exists(vocab_path):
            self.create_vocab(vocab_path)
        self.vocab = sorted(list(open(vocab_path, 'r').read().split('\n')))
        # ensure the padding token comes first so that it has index 0
        self.vocab = ['<pad>'] + [word for word in self.vocab if word != '<pad>']
        self.word2index = {word: i for i, word in enumerate(self.vocab)}  # type: Dict[str, int]
        self.index2word = {i: word for i, word in enumerate(self.vocab)}  # type: Dict[int, str]
        self.vocab_size = len(self.vocab)  # type: int

    def create_vocab(self, vocab_path: str) -> None:
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


if __name__ == "__main__":
    t = Tokenizer()
    dataset = Dataset(t)
    
    for i in range(15):
        x, y = dataset.get_training_data()
        yd = t.decode(y)
        xd = t.decode(x)
        print('x is', x)
        print('y is', y)
        print('decoded:')
        print('x is: ', xd)
        print('y is: ', yd)
        print()
    

    

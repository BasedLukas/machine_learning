from typing import Tuple, List
from multiprocessing import Pool
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
import pickle
from tqdm import tqdm


class SlidingWindowDataset(Dataset):
    def __init__(self, stories: list, params: dict):
        super(SlidingWindowDataset, self).__init__()
        self.seq_len = params['seq_len']
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        if os.path.exists(params['tokenized_data']):
            print('Loading tokenized data...')
            with open(params['tokenized_data'], 'rb') as f:
                self.tokenized_data = pickle.load(f)
        else:
            print('Tokenizing and building dataset...')
            if params.get('max_dataset_tokens', None):
                print(f'Using only {params["max_dataset_tokens"]:,} tokens')
                stories = stories[:params['max_dataset_tokens']]
            self.tokenized_data = self.tokenize_stories(stories)
            with open(params['tokenized_data'], 'wb') as f:
                pickle.dump(self.tokenized_data, f)
            print(f'Done building dataset with {len(self.tokenized_data):,} tokens')
            
        if params.get('max_dataset_tokens', None):
            self.tokenized_data = self.tokenized_data[:params['max_dataset_tokens']]
        print(f'Loaded tokenized data with {len(self.tokenized_data):,} tokens')

    def tokenize_stories(self, stories: list) -> list:
        tokenized_data = []
        for story in tqdm(stories):
            story =  " " + story
            tokenized_story = self.tokenizer.encode(story, add_special_tokens=False)
            tokenized_data.extend(tokenized_story)
        return tokenized_data

    def __len__(self) -> int:
        return len(self.tokenized_data) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenized_data[idx: idx + self.seq_len + 1]
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        return x, y


if __name__ == '__main__':

    tinystories_dataset = load_dataset("roneneldan/TinyStories")
    tinystories_dataset = load_dataset("roneneldan/TinyStories")
    params_train = {
        "seq_len": 128,
        'tokenized_data': 'tokenized_data_train_small.pkl',
        "max_dataset_tokens": 50_000_000
    }
    params_val = {
        "seq_len": 128,
        'tokenized_data': 'tokenized_data_val_small.pkl',
        "max_dataset_tokens": 1_000
    }

    val_dataset = SlidingWindowDataset(
        tinystories_dataset["validation"]["text"], 
        params_val
    )
    train_dataset = SlidingWindowDataset(
        tinystories_dataset["train"]["text"],
        params_train
    )

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for x, y in train_loader:
        print(x.shape, y.shape)
        break
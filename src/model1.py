import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import tiktoken



# Load the tinystories dataset
tinystories_dataset = load_dataset("roneneldan/TinyStories")
enc = tiktoken.get_encoding("cl100k_base")



# Define a function to tokenize text
def tokenize_text(text):
    return enc.encode(text)


class TinyStoriesDataset(Dataset):
    def __init__(self, texts):
        # self.tokenized_texts = [tokenize_text(text) for text in texts]
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # x is all tokens except the last, y is all tokens except the first
        x = self.texts[idx][:-1]
        y = self.texts[idx][1:]
        x = tokenize_text(x)
        y = tokenize_text(y)
        return x, y

# Create a dataset
train_texts = tinystories_dataset['train']['text'] #iterable of story strings
train_dataset = TinyStoriesDataset(train_texts)

# Create a DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for x, y in train_dataloader:
    print(x.shape)
    print(y.shape)
    print(x)
    print(y)
    break
print(x.shape)
print(y.shape)
print(x)
print(y)
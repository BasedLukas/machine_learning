from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import Transformer  # Ensure this is the correct path to your model file
from data import SlidingWindowDataset  # Ensure this is the correct path to your dataset file
import os


# Parameters (modify these as per your needs)
params = {
    'batch_size': 1,
    'vocab_size': 30522,
    'seq_len': 64,
    'embedding_size': 1024,
    'n_heads': 16,
    'head_size': 64,
    'n_layers': 8,
    'tokenized_data': 'tokenized_data_val.pkl', 
    'saved_model_name': 'model_v2.pt',
}


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
def load_model(model_path: str, params: dict) -> nn.Module:
    model = Transformer(params).to('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Inference Data Loading
def load_data(tokenized_data_path: str, params: dict) -> DataLoader:
    dataset = SlidingWindowDataset([], params) 
    return DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)

# Generate Text Function
def generate_text(model: nn.Module, dataloader: DataLoader, max_length: int = 20, temperature: float = 1.0, top_k: int = 50) -> Tuple[str, str]:
    x, _ = next(iter(dataloader))
    x = x.to(device)
    input_ids = x[0:1]
    original_input_ids = input_ids.clone()
    generated_tokens = torch.tensor([], device=device).long()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]

            # Apply temperature scaling and top-k sampling
            predictions = predictions / temperature
            top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), top_k, sorted=True)
            indices_to_sample = torch.multinomial(top_k_probs, 1)
            predicted_token_id = torch.gather(top_k_indices, 1, indices_to_sample)

            # Collect generated tokens
            generated_tokens = torch.cat([generated_tokens, predicted_token_id], dim=1)
            input_ids = torch.cat([input_ids, predicted_token_id], dim=-1)[:, 1:]
            input_ids = input_ids[:, -64:]

    # Decode the texts
    original_text = dataloader.dataset.tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
    generated_text = dataloader.dataset.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return original_text, generated_text


def generate_from_prompt(model: nn.Module, tokenizer: BertTokenizer, prompt: str, max_length: int = 20, temperature: float = 1.0, top_k: int = 50) -> str:
    # Prepare the prompt
    prompt = "."*64 + prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    input_ids = input_ids[:, -64:]
    generated_tokens = torch.tensor([], device=device).long()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]

            # Apply temperature scaling and top-k sampling
            predictions = predictions / temperature
            top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), top_k, sorted=True)
            indices_to_sample = torch.multinomial(top_k_probs, 1)
            predicted_token_id = torch.gather(top_k_indices, 1, indices_to_sample)

            # Collect generated tokens
            generated_tokens = torch.cat([generated_tokens, predicted_token_id], dim=1)
            input_ids = torch.cat([input_ids, predicted_token_id], dim=-1)
            input_ids = input_ids[:, -64:]  # Keep only the last 64 tokens

    # Decode the texts
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text


# Main Execution
if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(params['saved_model_name'], params).to(device)
    dataloader = load_data(params['tokenized_data'], params)


    generation_params = [
        {'temperature': 1.0, 'top_k': 500},
        # {'temperature': 0.7, 'top_k': 50},
        # {'temperature': 0.5, 'top_k': 50},
        {'temperature': 1, 'top_k': 1000},
        # {'temperature': 0.7, 'top_k': 100},
        # {'temperature': 0.5, 'top_k': 100},
        {'temperature': 1, 'top_k': 50},
        # {'temperature': 0.7, 'top_k': 5},
        # {'temperature': 0.5, 'top_k': 5},
    ]


    for generation_param in generation_params:
        original_text, generated_text = generate_text(model, dataloader, max_length=100, **generation_param)
        print(f"Generation Parameters: {generation_param}")
        print(f"Original Text: {original_text}\nGenerated Text: {generated_text}\n")

    prompt = "once apon a time there was a "
    generated_text = generate_from_prompt(model, tokenizer, prompt, max_length=100, top_k=75)
    print(f"Prompt: {prompt}\nGenerated Text: {generated_text}\n") 
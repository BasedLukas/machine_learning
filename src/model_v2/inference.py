from typing import Tuple, List
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from model import Transformer  # Ensure this is the correct path to your model file
from data import SlidingWindowDataset  # Ensure this is the correct path to your dataset file
import os


# Parameters (modify these to match your model, set batch to 1)
params = {
    'batch_size': 1,
    'vocab_size': 30522,
    'seq_len': 64,
    'embedding_size': 256,
    'n_heads': 16,
    'head_size': 16,
    'n_layers': 4, # transformer blocks
    'tokenized_data':'tokenized_data_train_small.pkl', # name of the tokenized data file
    'max_dataset_tokens': 10_000_000, # how many tokens to use from the dataset
    'epochs': 1,
    'min_items_update': 5000, # how often to update the weights
    'warmup_steps': 50,
    'lr': 0.0001,
    'patience': 25,
    'saved_model_name': 'model_v2.pt',
    'use_wandb': 'disabled' # 'disabled', 'online', 'offline'
}


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
def load_model(model_path: str, params: dict) -> nn.Module:
    model = Transformer(params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Inference Data Loading
def load_data(params: dict) -> DataLoader:
    dataset = SlidingWindowDataset([], params) 
    return DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)


def process_logits(predictions: torch.Tensor, temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
    # Apply temperature scaling and top-k sampling
    predictions = predictions / temperature
    top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), top_k, sorted=True)
    indices_to_sample = torch.multinomial(top_k_probs, 1)
    predicted_token_id = torch.gather(top_k_indices, 1, indices_to_sample)
    return predicted_token_id

import torch

def process_logits(predictions: torch.Tensor, temperature: float = 1.0, top_k: int = 50, top_p: float = None, greedy: bool = False) -> torch.Tensor:
    """
    Process logits with different sampling methods.

    Args:
    predictions (torch.Tensor): The logits from the model.
    temperature (float): Temperature for scaling.
    top_k (int): Number of top tokens to consider for sampling.
    top_p (float): Cumulative probability threshold for nucleus sampling.
    greedy (bool): If true, use greedy sampling.

    Returns:
    torch.Tensor: Predicted token ids.
    """
    # Apply temperature scaling
    predictions = predictions / temperature

    if greedy:
        # Greedy sampling: choose the highest probability token
        _, predicted_token_id = torch.max(predictions, dim=-1, keepdim=True)
    else:
        if top_p is not None:
            # Top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(predictions, descending=True)
            cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_keep = sorted_indices[~sorted_indices_to_remove]
            top_k_indices = indices_to_keep[:, :top_k]
        else:
            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), top_k, sorted=True)

        # Sampling from the filtered set of indices
        top_k_probs = torch.softmax(predictions.gather(-1, top_k_indices), dim=-1)
        indices_to_sample = torch.multinomial(top_k_probs, 1)
        predicted_token_id = top_k_indices.gather(-1, indices_to_sample)

    return predicted_token_id


# Generate Text Function
def generate_text(
        model: nn.Module, 
        dataloader: DataLoader, 
        max_length: int = 20, 
        temperature: float = 1.0, 
        top_k: int = 50,
        top_p: float = None,
        greedy: bool = False
    ) -> Tuple[str, str]:
    x, _ = next(iter(dataloader))
    x = x.to(device)
    input_ids = x[0:1]
    original_input_ids = input_ids.clone()
    generated_tokens = torch.tensor([], device=device).long()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]

            # Process the logits
            predicted_token_id = process_logits(predictions, temperature, top_k)

            # Collect generated tokens
            generated_tokens = torch.cat([generated_tokens, predicted_token_id], dim=1)
            input_ids = torch.cat([input_ids, predicted_token_id], dim=-1)[:, 1:]
            input_ids = input_ids[:, -128:]

    # Decode the texts
    original_text = dataloader.dataset.tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
    generated_text = dataloader.dataset.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return original_text, generated_text


def beam_search(model: nn.Module, input_ids: torch.Tensor, beam_width: int, max_length: int) -> List[Tuple[torch.Tensor, float]]:
    beams = [(input_ids, 0.0)]  # (sequence, score)
    model_max_length = params['seq_len'] 
    for step in range(max_length - input_ids.size(1)):
        new_beams = []
        for seq, score in beams:
            if seq.size(1) >= max_length:
                new_beams.append((seq, score))
                continue

            # Truncate sequence if it exceeds the model's max length
            if seq.size(1) > model_max_length:
                seq = seq[:, -model_max_length:]

            with torch.no_grad():
                output = model(seq)
                logits = output[:, -1, :]
                probs = F.softmax(logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

            for i in range(beam_width):
                next_token_id = top_indices[:, i]
                next_score = score + torch.log(top_probs[:, i]).item()
                new_beam = (torch.cat([seq, next_token_id.unsqueeze(-1)], dim=1), next_score)
                new_beams.append(new_beam)

        # Keep only the best beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    return beams

def generate_text_with_beam_search(
        model: nn.Module, 
        dataloader: DataLoader, 
        max_length: int = 20, 
        beam_width: int = 5
    ) -> Tuple[str, str]:
    x, _ = next(iter(dataloader))
    x = x.to(device)
    input_ids = x[0:1]
    original_input_ids = input_ids.clone()

    beams = beam_search(model, input_ids, beam_width, max_length)
    best_sequence, _ = max(beams, key=lambda x: x[1])

    # Decode the texts
    original_text = dataloader.dataset.tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
    generated_text = dataloader.dataset.tokenizer.decode(best_sequence[0], skip_special_tokens=True)
    return original_text, generated_text

# Main Execution
if __name__ == '__main__':

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = load_model(params['saved_model_name'], params).to(device)
    dataloader = load_data(params)

    ### WORK IN PROGRESS ###
    # beam_params = [
    #     {'beam_width': 1},
    #     {'beam_width': 5},
    #     {'beam_width': 10},
    #     {'beam_width': 20},
    #     {'beam_width': 50},
    # ]
    # for beam_param in beam_params:
    #     original_text, generated_text = generate_text_with_beam_search(model, dataloader, max_length=100, **beam_param)
    #     print(f"Beam Search Parameters: {beam_param}")
    #     print(f"Original Text: {original_text}\nGenerated Text: {generated_text}\n")

    generation_params = [
        {'temperature': 1.0, 'top_k': 500},
        {'temperature': 1, 'top_k': 1000},
        {'temperature': 1, 'top_k': 50},
        {'temperature': 1, 'top_k': 5, 'greedy': True},
        {'temperature': 0.5, 'top_k': 50, 'greedy': False, 'top_p': 0.9},
        {'temperature': 0.8, 'top_k': 200, 'greedy': False, 'top_p': 0.9},
    ]
    for generation_param in generation_params:
        original_text, generated_text = generate_text(model, dataloader, max_length=100, **generation_param)
        print(f"Generation Parameters: {generation_param}")
        print(f"Original Text: {original_text}\nGenerated Text: {generated_text}\n")

from typing import List, Tuple, Dict, Set, Union, Any, cast, Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import  DataLoader
from data import SlidingWindowDataset
from model import Transformer
import os
from dotenv import load_dotenv
import wandb

### PARAMS ###
params = {
    'batch_size': 40,
    'vocab_size': 30522,
    'seq_len': 64,
    'embedding_size': 1024,
    'n_heads': 16,
    'head_size': 64,
    'n_layers': 8, # transformer blocks
    'tokenized_data':'tokenized_data_train.pkl', # name of the tokenized data file
    'max_dataset_tokens': 10_000_000, # how many tokens to use from the dataset
    'epochs': 1,
    'min_items_update': 5000, # how often to update the weights
    'warmup_steps': 20,
    'lr': 0.0001,
    'patience': 5,
    'saved_model_name': 'model_v2.pt',
}
val_params = params.copy()
val_params['max_dataset_tokens'] = 1_000 # how many tokens to use for validation
assert params['embedding_size'] / params['n_heads'] == params['head_size']

### WANDB ###
load_dotenv()
wandb_api_key = os.getenv("WANDB")
wandb.login(key=wandb_api_key)
wandb.init(
    project="transformer",
    config=params,
)

### OBJECTS ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(params).to(device).train()
dataset = SlidingWindowDataset([],params)
dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
val_dataset = SlidingWindowDataset([],val_params)
val_dataloader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params['patience'], verbose=True, min_lr=1e-9, factor=0.5)
loss_fn = nn.CrossEntropyLoss()

def warmup():
    pass

def generate_text(max_length=20, temperature=1.0, top_k=50)->Tuple[str, str]:
    model.eval()
    x, _ = next(iter(val_dataloader))
    x = x.to(device)
    input_ids = x[0:1]  # Shape: [1, seq_len]
    original_input_ids = input_ids.clone()  # Keep a copy of the original input_ids
    generated_tokens = torch.tensor([], device=device).long()

    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            predictions = outputs[:, -1, :]

            # Apply temperature scaling
            predictions = predictions / temperature

            # Top-k sampling
            top_k_probs, top_k_indices = torch.topk(torch.softmax(predictions, dim=-1), top_k, sorted=True)
            indices_to_sample = torch.multinomial(top_k_probs, 1)
            predicted_token_id = torch.gather(top_k_indices, 1, indices_to_sample)

            # Collect the generated tokens
            generated_tokens = torch.cat([generated_tokens, predicted_token_id], dim=1)

            # Append the predicted token to the input for the next iteration
            input_ids = torch.cat([input_ids, predicted_token_id], dim=-1)[:, 1:]

    # Decode the texts
    original_text = val_dataloader.dataset.tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
    generated_text = val_dataloader.dataset.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    model.train()
    return original_text, generated_text


def validation()->float:
    model.eval() 
    total_loss = 0
    total_items = 0

    with torch.no_grad(): 
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs.view(-1, params['vocab_size']), y.view(-1))
            total_loss += loss.item()
            total_items += x.size(0)

    avg_loss = total_loss / total_items
    model.train() 
    return avg_loss


def update_weights(avg_loss):
    global total_number_of_items_processed
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step(avg_loss)

    val_loss = validation()
    original_text, generated_text = generate_text(max_length=35, top_k=10)

    #create wandb table plot of avg_loss, val_loss, over total_number_of_items_processed
    wandb.Table(data=[[total_number_of_items_processed, avg_loss, val_loss]], 
                columns=["total_number_of_items_processed", "avg_loss", "val_loss"])
    wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "original_text": original_text, "generated_text": generated_text})
    
    global best_loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), params['saved_model_name'])
        wandb.log({'saved_new_best_model': best_loss})
    
    total_n_items = len(dataloader.dataset)
    print(f'\niter: {(total_number_of_items_processed/total_n_items)*100}, \nTrain Loss: {avg_loss}, \nVal Loss: {val_loss}')
    print(generated_text)


if params.get('warmup_steps') is not None:
    warmup()


# Main training loop
best_loss = float('inf')
total_number_of_items_processed = 0

for epoch in range(params['epochs']):
    items_processed = 0
    accumulated_loss = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        outputs = model(x)
        loss = loss_fn(outputs.view(-1, params['vocab_size']), y.view(-1))
        loss.backward()
        loss = loss.item()

        accumulated_loss += loss
        items_processed += x.size(0)
        total_number_of_items_processed += x.size(0)

        if items_processed >= params['min_items_update']:
            avg_loss = accumulated_loss / items_processed
            update_weights(avg_loss)
            items_processed = 0
            accumulated_loss = 0


    if items_processed > 0:
        avg_loss = accumulated_loss / items_processed
        update_weights(avg_loss)


  
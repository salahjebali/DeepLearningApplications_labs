import torch
import torch.nn as nn
from torch.nn import functional as F


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Set your GPT model in evaluation mode
m.eval()

def calculate_perplexity(model, dataset):
    total_loss = 0.0
    total_words = 0

    with torch.no_grad():
        for k in range(len(dataset)):
            X, y = get_batch('val')
            X = X.to(device)
            y = y.to(device)
          
            # Forward pass to get logits
            logits, loss = model(X, y)

            total_loss += loss.item() * y.numel()
            total_words += y.numel()

    # Calculate perplexity
    perplexity = torch.exp(total_loss / torch.tensor(total_words))

    return perplexity
import torch
from torch import nn
import numpy as np
from typing import Any, Union, Iterator, Generator
from nn_functions import *
from data_augmentation import *

def train_step(
        batch:tuple[torch.Tensor, torch.Tensor], 
        model:nn.Module, 
        criterion:nn.CrossEntropyLoss, 
        optimizer:torch.optim.AdamW, 
        lr_scheduler: Union[Generator, Iterator],
        AUGMENTATION_PIPELINE: Any, 
        device: str | torch.device):
    model.train()
    lr = next(lr_scheduler)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    inputs, targets = batch
    inputs, targets = augment_image_batch(inputs.to(device), AUGMENTATION_PIPELINE), targets.to(device)
    
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss: torch.Tensor = criterion(logits, targets)
    acc = accuracy(logits, targets)
    loss.backward()
    
    optimizer.step()
    return history(loss.item(), acc, lr, per_layer_grad_norm(model)) # type: ignore # TODO: change grad norm argument

# --- EVALUATION ---
@torch.no_grad()
def evaluate(
    model: nn.Module, 
    test_loader: torch.utils.data.DataLoader, 
    criterion: nn.CrossEntropyLoss, 
    device: str | torch.device
    ):
    model.eval()
    total_loss, total_acc, total_count = 0, 0, 0
    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        acc = accuracy(logits, targets)
        
        total_loss += loss.item() * len(inputs)
        total_acc += acc * len(inputs) # type: ignore
        total_count += len(inputs)
    
    avg_loss = total_loss / total_count
    avg_acc = total_acc / total_count
    return {"avg_loss": avg_loss, "avg_accuracy": avg_acc}
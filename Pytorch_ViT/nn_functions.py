import torch
from torch import nn
import numpy as np
import math
from typing import Union, Iterator, NamedTuple

@torch.no_grad()
def per_layer_grad_norm(model:nn.Module) -> dict:
    return {k: torch.linalg.norm(v.grad).item() for k, v in list(model.named_parameters()) if hasattr(v, "requires_grad") and (v.requires_grad == True)}

def summery_norm(parameter_dict:dict):
    grad_norm = np.array([v for k, v in parameter_dict.items()])
    return grad_norm.mean(), grad_norm.std()

@torch.no_grad()
def accuracy(logits:torch.Tensor, targets:torch.Tensor, return_pred_class:bool=False) -> Union[tuple, float]:
    logits_argmax = logits.argmax(-1)
    mean_accuracy = (logits_argmax == targets).to(torch.float32).mean().item()
    out = (mean_accuracy, logits_argmax) if return_pred_class else mean_accuracy
    return out

def imshow(img_tensor: torch.Tensor, title: str):
    import matplotlib.pyplot as plt
    """Helper function to un-normalize and display an image tensor."""
    img_tensor = img_tensor / 2 + 0.5  # Un-normalize from [-1, 1] to [0, 1]
    npimg = img_tensor.numpy()
    plt.figure(figsize=(5, 5), frameon=False)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.axis('off')
    plt.show()


def cos_anneal_lr_scheduler_gen(
    warmup_steps: int, total_anneal_steps: int, 
    lr_at_warmup_start: float, lr_at_cosine_start: float, lr_at_cosine_end: float
) -> Iterator[float]:
    """
    Generates learning rates: linear warmup then cosine annealing.  
    Note: total_anneal_steps is total_steps - warmup_steps
    """

    # Linear warmup phase
    for step in range(warmup_steps):
        progress = (step + 1) / warmup_steps
        current_lr = lr_at_warmup_start + progress * (lr_at_cosine_start - lr_at_warmup_start)
        yield current_lr

    # Cosine annealing phase
    for step in range(total_anneal_steps):
        cos_inner = math.pi * step / total_anneal_steps
        current_lr = lr_at_cosine_end + 0.5 * (lr_at_cosine_start - lr_at_cosine_end) * (1 + math.cos(cos_inner))
        yield current_lr

class History(NamedTuple):
    loss: list = []
    accuracy: list = []
    LR: list = []
    gradient_norm: list = []

    def __call__(self, loss:float, accuracy:float, lr:float, grad_norms:dict):
        self.loss.append(loss)
        self.accuracy.append(accuracy)
        self.LR.append(lr)
        self.gradient_norm.append(grad_norms)
        grad_norm_mean, grad_norm_std = summery_norm(grad_norms)
        list_to_fit = [
            f"Loss: {loss:.4f}",
            f"Acc: {accuracy:.4f}",
            f"LR: {lr:.6f}",
            f"grad norm mean: {grad_norm_mean:.3f}",
            f"grad norm std: {grad_norm_std:.3f}"]
        return " | ".join(list_to_fit)
    
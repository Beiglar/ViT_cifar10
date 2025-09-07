import torch
import random
import numpy as np
import math
from typing import Callable, Dict

# --- Augmentation Functions ---

def identity(images: torch.Tensor) -> torch.Tensor:
    """Returns the original images, unchanged."""
    return images

def random_horizontal_flip(images: torch.Tensor) -> torch.Tensor:
    """Flips images horizontally with a 50% probability."""
    if random.random() < 0.5:
        return torch.flip(images, dims=[3])
    return images

def add_normal_noise(images: torch.Tensor, noise_scale: float = 0.04) -> torch.Tensor:
    """Adds Gaussian noise to images."""
    noise = torch.randn_like(images)
    noise = noise - noise.min()
    noise = noise / noise.max()
    image = (1.0 - noise_scale) * images + noise_scale * noise
    return image

def add_uniform_noise(images: torch.Tensor, noise_scale: float = 0.05) -> torch.Tensor:
    """Adds uniform noise to images."""
    return (1.0 - noise_scale) * images + noise_scale * torch.rand_like(images)

def bernoulli_mask(images: torch.Tensor, keep_prob: float = 0.8) -> torch.Tensor:
    """Applies a random binary mask to images, setting some pixels to 0."""
    mask = torch.bernoulli(torch.full_like(images, keep_prob))
    return images * mask

def color_channel_flip(images: torch.Tensor) -> torch.Tensor:
    """Applies a single, random permutation of color channels to a batch of images."""
    # Note: This applies the *same* permutation to all images in the batch.
    permutation = torch.randperm(3)
    return images[:, permutation, :, :]

# --- Augmentation Pipeline Configuration ---
# Based on the logic from the original code, the effective probabilities for each
# distinct transformation have been calculated and are defined explicitly here.
# This makes the configuration much clearer and easier to modify.

# Original logic breakdown:
# Prob(Noise) = 0.6 * 0.8 = 0.48
# Prob(Color Flip) = 0.6 * 0.15 = 0.09
# Prob(K-Means Placeholder) = 0.6 * 0.05 = 0.03
# Prob(Identity) = 0.4
#
# The K-Means and Identity options both used a function that performed a 50/50
# horizontal flip or true identity. Total prob for this function = 0.03 + 0.40 = 0.43.
#
# Final calculated probabilities:
# - Normal Noise: 0.48 * 0.7 = 0.336
# - Uniform Noise: 0.48 * 0.2 = 0.096
# - Bernoulli Mask: 0.48 * 0.1 = 0.048
# - Color Flip: 0.09
# - Horizontal Flip: 0.43 * 0.5 = 0.215
# - True Identity: 0.43 * 0.5 = 0.215
# Sum: 0.336 + 0.096 + 0.048 + 0.09 + 0.215 + 0.215 = 1.0

def augment_image_batch(
    images: torch.Tensor,
    augmentations: Dict[Callable[[torch.Tensor], torch.Tensor], float]
) -> torch.Tensor:
    """
    Applies a random augmentation from the provided pipeline to each image in a batch.

    This version is robust and efficient, using boolean masking to apply transformations
    to relevant subsets of the image batch, avoiding complex and potentially buggy sorting.

    Args:
        images: A batch of images as a PyTorch tensor (B, C, H, W).
        augmentations: A dictionary mapping augmentation functions to their probabilities.

    Returns:
        A new tensor containing the augmented images.
    """
    aug_fns = list(augmentations.keys())
    probs = list(augmentations.values())
    
    # Verify that probabilities sum to 1.0
    assert math.isclose(sum(probs), 1.0), "Probabilities must sum to 1.0"
    
    # For each image in the batch, choose an augmentation function index based on probability
    B, C, H, W = images.shape
    choices = np.random.choice(len(aug_fns), size=(B,), p=probs)
    
    # Create an output tensor to store augmented images
    output_images = torch.empty_like(images)
    
    # Apply the chosen augmentation for each group of images that share the same choice
    for i, func in enumerate(aug_fns):
        # Create a boolean mask for the images that need this augmentation
        mask = (choices == i)
        
        # If any image was chosen for this augmentation, apply the function
        if mask.any():
            output_images[mask] = func(images[mask])
        
    return output_images
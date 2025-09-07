"""Minimal training runner (smoke test) for the modular JAX ViT package.

This script performs a single forward+backward step using one batch from CIFAR-10
to validate wiring between dataloader, augmentation, and model code.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax

from dataloader import get_dataloaders
from augmentation import augment_image_batch_vmap, AUGMENTATION_FNS, AUGMENTATION_PROBS
from nnx_modules import ViT, ViTConfig
from nnx_functions import train_step, compute_loss, get_labels


def to_jax(batch):
    images, labels = batch
    # images: torch tensor with shape (B, C, H, W)
    images = images.numpy()
    labels = labels.numpy()
    # keep as (B, C, H, W) for augmentation module
    return jnp.array(images), jnp.array(labels)


def main():
    train_loader, test_loader = get_dataloaders(batch_size=64)
    batch = next(iter(train_loader))
    images, labels = to_jax(batch)

    # Create model and optimizer (nnx API)
    config = ViTConfig(img_shape=(32, 32, 3), patch_size=1, num_classes=10, dim=32, depth=2, heads=4)
    model = ViT(config)
    rng = jax.random.PRNGKey(0)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=.001)
    ))

    # Apply augmentations (jax expects float arrays)
    aug_rng = jax.random.PRNGKey(1)
    images_aug = augment_image_batch_vmap(images, aug_rng, AUGMENTATION_FNS, AUGMENTATION_PROBS)

    # Run a single train step
    loss, acc, grads = train_step(model, optimizer, images_aug.swapaxes(1, -1), labels)

    print(f"Smoke test: loss={float(loss):.4f}, acc={float(acc):.4f}")


if __name__ == '__main__':
    main()

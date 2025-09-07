# Vision Transformer (ViT) Implementations for CIFAR-10

This repository contains two implementations of a hybrid Vision Transformer (ViT) model for image classification on the CIFAR-10 dataset. The implementations are provided in both JAX/Flax (using the modern `nnx` API) and PyTorch.

The core model architecture is a hybrid that uses a small Convolutional Neural Network (CNN) as a "patchifier" to extract feature maps, which are then flattened and fed into a standard Transformer encoder.

## Implementations

### 1. JAX/Flax (NNX)

This implementation uses JAX and Flax's new `nnx` module API, which offers a more familiar, object-oriented programming model similar to PyTorch while retaining JAX's performance benefits.

- **Main Notebook:** `JAX_ViT.ipynb`
  - This notebook contains the complete workflow: data loading, model definition, training loop, evaluation, and visualization of results.
- **Modular Code:** The core logic is broken down into modules inside the `JAX_ViT/` directory:
  - `nnx_modules.py`: Defines the ViT architecture, including the `Attention`, `TransformerBlock`, and `ViT` classes using `flax.nnx`.
  - `augmentation.py`: A custom, JAX-native data augmentation pipeline with functions for random flips, cutouts, color jitter, and more.
  - `dataloader.py`: Utilities for handling the data pipeline.
  - `train.py`: A minimal script to test that the model and data pipeline are wired correctly.

### 2. PyTorch

This implementation provides a more traditional approach using PyTorch.

- **Main Notebook:** `Pytorch_ViT.ipynb`
  - Contains the full pipeline for the PyTorch version of the model.
- **Modular Code:** The logic is organized in the `Pytorch_ViT/` directory:
  - `nn_modules.py`: Defines the ViT model using PyTorch's `nn.Module`.
  - `data_augmentation.py`: Data augmentation pipeline using `torchvision.transforms`.
  - `train_eval.py`: Contains the training and evaluation functions.
  - `train_model.py`: Script to run the full training process.

## Features

- **Hybrid ViT Architecture:** A CNN frontend acts as a feature extractor, replacing the standard linear patch projection.
- **Modern JAX:** The JAX version is built with `flax.nnx`, showcasing a new and more ergonomic API for model development in Flax.
- **Custom Augmentation:** The JAX implementation features a from-scratch, JIT-compatible data augmentation pipeline.
- **Training Utilities:** Includes learning rate scheduling (warmup-cosine decay), gradient clipping, and detailed logging.
- **Pre-trained Model:** A pre-trained PyTorch model state dictionary is provided (`ViT_CIFAR10_235,882.pt`).

## Setup and Usage

1.  **Create a virtual environment and activate it:**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

2.  **Install Dependencies:**
    The required packages are used by both projects. You can install them from the provided `requirements.txt`.
    ```bash
    pip install -r JAX_ViT/requirements.txt
    ```
    *Note: JAX installation can be platform-specific. If you have a compatible NVIDIA GPU, you may need to install a GPU-specific version of JAX. See the [official JAX installation guide](https://github.com/google/jax#installation) for details.*

3.  **Run the Notebooks:**
    The easiest way to explore the models is to run the Jupyter notebooks:
    - `JAX_ViT.ipynb`
    - `Pytorch_ViT.ipynb`

    These notebooks guide you through the entire process, from data loading to model training and evaluation.

## File Guide

- `JAX_ViT.ipynb`: Main notebook for the JAX/Flax implementation.
- `Pytorch_ViT.ipynb`: Main notebook for the PyTorch implementation.
- `JAX_ViT/`: Directory for the modularized JAX code.
- `Pytorch_ViT/`: Directory for the modularized PyTorch code.
- `ViT_CIFAR10_235,882.pt`: Pre-trained model weights for the PyTorch version.
- `ViT_train_info_*.log`: Log files generated during training sessions.
- `.gitignore`: Standard Python gitignore file.
- `README.md`: This file.

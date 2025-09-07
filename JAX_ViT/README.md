JAX_ViT — modularized Vision Transformer in JAX/Flax

This folder contains a modularized ViT implementation split into:
- dataloader.py — CIFAR-10 DataLoader helpers (torch-jax collate)
- augmentation.py — JAX-native augmentation utilities
- nnx_modules.py — ViT model components (Flax nnx)
- nnx_functions.py — training/eval helpers
- train.py — minimal smoke-test runner that performs one train step

Quick setup (Windows PowerShell):
# create a virtual env and install deps
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Run the smoke test:
# from the JAX_ViT directory
python train.py

Notes:
- JAX may require CPU/GPU-specific wheels; see https://github.com/google/jax for installation instructions.
- The smoke test only runs one train step to verify wiring; it is not a full training script.

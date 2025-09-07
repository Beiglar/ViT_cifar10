import math
import time
import logging
import datetime as dt
import matplotlib.pyplot as plt
from typing import Callable, Dict
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

from data_augmentation import *
from nn_functions import *
from nn_modules import *
from train_eval import *
# import torch._dynamo; torch._dynamo.config.suppress_errors = True

# CONFIGURATION
# --- Data Configuration ---
DATASET_PATH = Path(r"F:\deep_experiments\datasets\CIFAR10_data")
BATCH_SIZE = 512
N_CLASSES = 10

# --- Device Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on matmul
torch.backends.cudnn.allow_tf32 = True  # Allow TF32 on cudnn
torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
torch._C._jit_set_profiling_executor(True)
torch._C._jit_set_profiling_mode(True)

AUGMENTATION_PIPELINE: Dict[Callable[[torch.Tensor], torch.Tensor], float] = {
    add_normal_noise: 0.336,
    add_uniform_noise: 0.096,
    bernoulli_mask: 0.048,
    color_channel_flip: 0.09,
    random_horizontal_flip: 0.215,
    identity: 0.215,  # Note: K-Means is not implemented and its probability
                      # is absorbed into the identity/flip operations as in the original.
}

# DATA LOADING

# Define the transformation to normalize images into the range [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset path if it doesn't exist
DATASET_PATH.mkdir(parents=True, exist_ok=True)

# Load CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root=DATASET_PATH, train=True, download=False, transform=transform)
test_dataset = datasets.CIFAR10(root=DATASET_PATH, train=False, download=False, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# --- Fetch a batch of original images ---
original_images, _ = next(iter(train_loader))

# --- Apply the new augmentation pipeline ---
augmented_images = augment_image_batch(original_images, AUGMENTATION_PIPELINE)

# --- Display the results ---
# Show a grid of the first 16 original images
imshow(make_grid(original_images[:16], nrow=4), "Original Images")

# Show a grid of the first 16 augmented images
imshow(make_grid(augmented_images[:16], nrow=4), "Augmented Images")

print("Demonstration complete. Original and augmented image grids have been displayed.")

# Model & Optimizer Instantiation

# --- Model Hyperparameters ---
model = ViT(
    img_shape=(32, 32, 3),
    patch_size=1, # After CNN, feature map is 8x8. We treat each 1x1 pixel as a "patch".
    num_classes=N_CLASSES,
    dim=32,       # Embedding dimension
    depth=6,      # Number of transformer blocks
    heads=8,      # Number of attention heads
    mlp_ratio=4,
    dropout=0.1
).to(device)

# Doesn't work on my GPU
# # Use torch.compile for a significant speedup on PyTorch 2.0+
# if hasattr(torch, 'compile'):
#     print("Compiling the model...")
#     model = torch.compile(model)

parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {parameter_count:,}")
expected_untrained_loss = -math.log(1/N_CLASSES)
print(f"Expected untrained loss: {expected_untrained_loss:.4f}")

# --- Optimizer and Loss ---
NUM_EPOCHS = 40
num_batches = len(train_loader)
total_optim_steps = num_batches * NUM_EPOCHS
warmup_steps = int(0.1 * total_optim_steps)
total_anneal_steps = total_optim_steps - warmup_steps

lr_scheduler = cos_anneal_lr_scheduler_gen(
    warmup_steps=warmup_steps,
    total_anneal_steps=total_anneal_steps,
    lr_at_warmup_start=1e-5,
    lr_at_cosine_start=1e-3,
    lr_at_cosine_end=1e-5
)

loss_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.05) # Start LR is managed by scheduler

history = History()

log_name = f"ViT_train_info_{dt.datetime.now().date().isoformat()}"
logger = logging.getLogger(log_name)
logging.basicConfig(filename=f'{log_name}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')

# TRAINING LOOP
print("Starting training...")
num_train_batches = len(train_loader)
for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    for batch_idx, batch in enumerate(train_loader):
        history_report = train_step(batch, model, loss_criterion, optimizer, lr_scheduler, AUGMENTATION_PIPELINE, device)

        if batch_idx % 10 == 0:
            reports = [
                f"\rEpoch: {round(epoch + (batch_idx/num_train_batches), 2):.2f}/{NUM_EPOCHS}",
                history_report
                ]
            logger.info(" -- ".join(reports))

        if batch_idx % 20 == 0:
            print(*reports, sep=' | ', end='', flush=True)

    Epoch_Report = [
        f"\r>>> Epoch {round(epoch + (batch_idx/num_train_batches))} complete", 
        f"Epoch lasted {round(time.time() - epoch_start_time)} seconds <<<\n"]
    logger.info(" --- ".join(Epoch_Report))

print("\nTraining finished.")
# Save the model (use the compiled model if available)
torch.save(model.state_dict(), f"ViT_CIFAR10_{parameter_count:,}.pt")

# To evaluate, load the state dict into a non-compiled model instance
eval_model = ViT(img_shape=(32, 32, 3), patch_size=1, num_classes=10, dim=32, depth=6, heads=8).to(device)
eval_model.load_state_dict(torch.load(f"ViT_CIFAR10_{parameter_count:,}.pt")) # ctrl c-v model name if loading later
eval_results = evaluate(eval_model, test_loader, loss_criterion, device)
print(f"Evaluation Results -> Avg Loss: {eval_results['avg_loss']:.4f}, Avg Accuracy: {eval_results['avg_accuracy']:.4f}")

clean_dict = {}
for key in history.gradient_norm[0].keys():
    current_list = []
    for snap in history.gradient_norm:
        current_list.append(snap[key])
    clean_dict[key] = current_list

def plot(k:str):
    plt.figure(figsize=(6, 4), frameon=False)
    plt.plot(history.__getattribute__(k), '.', alpha=.8, ms=1)
    plt.title(k.replace('_', ' ').capitalize())
    plt.show()

for k in history.__annotations__:
    if k != 'gradient_norm':
        plot(k)

rows = round(len(clean_dict)**.5)
dm = divmod(len(clean_dict), rows)
cols = dm[0]+1 if dm[1] != 0 else dm[0]

fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
fig.frameon = False
fig.tight_layout(pad=1.2)
for i, (ax, (k, v)) in enumerate(zip(axes.flat, clean_dict.items())):
    ax.plot(v, '.', ms=.5)
    if k.endswith('.weight'): k = k.replace('.weight', '.w')
    elif k.endswith('.bias'): k = k.replace('bias', '.b')
    ax.set_title(k, y=.96, fontsize=6)
    ax.axis('off')
"""
Data loading and preprocessing for CIFAR-10 dataset.
"""
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- Data Configuration ---
DATASET_PATH = Path(r"F:\deep_experiments\datasets\CIFAR10_data")
BATCH_SIZE = 512
N_CLASSES = 10

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_dataloaders(batch_size: int = BATCH_SIZE) -> tuple[DataLoader, DataLoader]:
    """
    Creates and returns the CIFAR-10 train and test dataloaders.
    Handles dataset download if not found locally.
    """
    if os.path.exists(DATASET_PATH):
        download_dataset = False
        local_path = DATASET_PATH
    else:
        # Fallback for different environment
        local_path = Path(r"CIFAR10_data")
        download_dataset = True

    # Create the dataset path if it doesn't exist
    local_path.mkdir(parents=True, exist_ok=True)

    # Load CIFAR-10 datasets
    train_dataset = datasets.CIFAR10(root=local_path, train=True, download=download_dataset, transform=transform)
    test_dataset = datasets.CIFAR10(root=local_path, train=False, download=download_dataset, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader

def display_cifar_samples(train_loader: DataLoader, num_samples=16, rows=4, cols=4):
    """Displays a grid of sample images from the CIFAR-10 dataset."""
    images, labels = next(iter(train_loader))

    # Convert from NCHW to NHWC format and from [-1,1] to [0,1] range
    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))  # NCHW -> NHWC
    images = (images + 1) / 2.0  # [-1,1] -> [0,1]

    cifar_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck']

    fig, axes = plt.subplots(rows, cols, figsize=(4, 4))
    fig.frameon = False
    fig.tight_layout(pad=1.2)

    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images[i])
            ax.set_title(cifar_classes[labels[i]], y=.96, fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    plt.suptitle("CIFAR-10 Sample Images", y=1.02, fontsize=10)
    plt.show()

def get_labels(data_loader: DataLoader) -> np.ndarray:
    """Extracts all labels from a DataLoader into a numpy array."""
    return np.array([l for batch in data_loader for l in  batch[1].numpy().tolist()])

def plot_hist(x: np.ndarray, title:str|None=None):
    """Plots a histogram of the given data."""
    plt.figure(figsize=(4, 2), frameon=False)
    plt.hist(x, rwidth=.5)
    if title is not None:
        plt.suptitle(title)
    plt.show()

if __name__ == '__main__':
    train_loader, test_loader = get_dataloaders()

    print("Displaying sample images...")
    display_cifar_samples(train_loader)

    print("\nPlotting class distributions...")
    train_labels = get_labels(train_loader)
    test_labels = get_labels(test_loader)
    plot_hist(train_labels, "Train Class Distribution")
    plot_hist(test_labels, "Test Class Distribution")

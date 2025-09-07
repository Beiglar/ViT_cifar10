import jax
import jax.numpy as jnp
import numpy as np
import math
from functools import partial
from typing import TypeAlias, Union, Dict, Callable, Tuple
import jaxlib.xla_extension
import matplotlib.pyplot as plt

# --- Type Hinting ---
Array: TypeAlias = jax.Array
PRNGKey: TypeAlias = Union[jaxlib.xla_extension.ArrayImpl, jax.random.PRNGKey] # type: ignore

# --- Augmentation Functions ---

def identity(images: Array, rng: PRNGKey) -> Array:
    """Returns the images unchanged."""
    return images

def horizontal_flip(images: Array, rng: PRNGKey) -> Array:
    """Flips images horizontally."""
    return jnp.flip(images, axis=3)

def add_normal_noise(images: Array, noise_scale: float = 0.25, rng: PRNGKey = jax.random.key(0)) -> Array:
    """Adds Gaussian noise to images."""
    noise = jax.random.normal(rng, images.shape)
    noise = noise - noise.min()
    noise = noise / noise.max()
    image = (1.0 - noise_scale) * images + noise_scale * noise
    return image

def add_uniform_noise(images: Array, noise_scale: float = 0.05, rng: PRNGKey = jax.random.key(0)) -> Array:
    """Adds uniform noise to images."""
    return (1.0 - noise_scale) * images + noise_scale * jax.random.uniform(rng, images.shape)

def bernoulli_2Dmask(images: Array, keep_prob: float = 0.8, rng: PRNGKey = jax.random.key(0)) -> Array:
    """Applies a random binary mask to the spatial dimensions of images."""
    mask = jax.random.bernoulli(rng, keep_prob, shape=images.shape[-2:])
    return images * mask

def bernoulli_3Dmask(images: Array, keep_prob: float = 0.95, rng: PRNGKey = jax.random.key(0)) -> Array:
    """Applies a random binary mask across all dimensions of the images."""
    mask = jax.random.bernoulli(rng, keep_prob, shape=images.shape)
    return images * mask

def color_channel_flip(images: Array, rng: PRNGKey = jax.random.key(0)) -> Array:
    """Applies a single, random permutation of color channels to a batch of images."""
    permutation = jax.random.permutation(rng, 3)
    return images[:, permutation, :, :]

def random_color_jitter(
    images: Array, rng: PRNGKey, brightness_strength: float = 0.2, contrast_strength: float = 0.2,
) -> Array:
    """Applies random brightness and contrast adjustments."""
    batch_size = images.shape[0]
    rng_bright, rng_contrast = jax.random.split(rng)
    brightness_shifts = jax.random.uniform(
        rng_bright, shape=(batch_size, 1, 1, 1), minval=-brightness_strength, maxval=brightness_strength
    )
    jittered_images = images + brightness_shifts
    means = jnp.mean(jittered_images, axis=(2, 3), keepdims=True)
    contrast_factors = jax.random.uniform(
        rng_contrast, shape=(batch_size, 1, 1, 1), minval=1.0 - contrast_strength, maxval=1.0 + contrast_strength
    )
    jittered_images = means + contrast_factors * (jittered_images - means)
    return jnp.clip(jittered_images, -1.0, 1.0)

def random_cutout(images: Array, rng: PRNGKey, patch_size: int = 8) -> Array:
    """Randomly masks out a square patch in each image."""
    batch_size, _, h, w = images.shape
    rng_y, rng_x = jax.random.split(rng)
    center_y = jax.random.randint(rng_y, shape=(batch_size,), minval=0, maxval=h)
    center_x = jax.random.randint(rng_x, shape=(batch_size,), minval=0, maxval=w)
    half_patch = patch_size // 2
    y0 = jnp.clip(center_y - half_patch, 0, h)
    y1 = jnp.clip(center_y + half_patch, 0, h)
    x0 = jnp.clip(center_x - half_patch, 0, w)
    x1 = jnp.clip(center_x + half_patch, 0, w)

    def apply_cutout_to_one_image(image, y0, y1, x0, x1):
        yy, xx = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing='ij')
        mask = (yy >= y0) & (yy < y1) & (xx >= x0) & (xx < x1)
        return image * (1.0 - jnp.expand_dims(mask, axis=0))

    return jax.vmap(apply_cutout_to_one_image)(images, y0, y1, x0, x1)


# --- Augmentation Pipeline Setup ---

AUGMENTATION_PIPELINE: Dict[Callable, float] = {
    identity: 0.25,
    horizontal_flip: 0.10,
    add_normal_noise: 0.15,
    add_uniform_noise: 0.05,
    bernoulli_2Dmask: 0.10,
    bernoulli_3Dmask: 0.03,
    color_channel_flip: 0.02,
    random_color_jitter: 0.15,
    random_cutout: 0.15,
}

# Normalize probabilities to ensure they sum to 1
total_prob = sum(AUGMENTATION_PIPELINE.values())
AUGMENTATION_PIPELINE = {k: v / total_prob for k, v in AUGMENTATION_PIPELINE.items()}

AUGMENTATION_FNS = tuple(AUGMENTATION_PIPELINE.keys())
AUGMENTATION_PROBS = tuple(AUGMENTATION_PIPELINE.values())
assert math.isclose(sum(AUGMENTATION_PROBS), 1.0)

# --- Vmapped Augmentation Application ---

def apply_random_augmentation_to_one_image(
    image: Array, rng: PRNGKey, aug_fns: Tuple[Callable, ...], probs: jnp.ndarray
) -> Array:
    """Selects and applies a single augmentation function to one image."""
    choice_rng, fn_rng = jax.random.split(rng)
    aug_index = jax.random.choice(choice_rng, a=len(aug_fns), p=probs)

    def create_wrapped_fn(func):
        def wrapped_fn(img, key):
            # The augmentation functions expect a batch dimension, so we add and remove it.
            img_batched = jnp.expand_dims(img, axis=0)
            augmented_batched = func(img_batched, rng=key)
            return jnp.squeeze(augmented_batched, axis=0)
        return wrapped_fn

    wrapped_fns = [create_wrapped_fn(f) for f in aug_fns]
    return jax.lax.switch(aug_index, wrapped_fns, image, fn_rng)

@partial(jax.jit, static_argnames=('aug_fns',))
def augment_image_batch_vmap(
    images: Array, rng: PRNGKey, aug_fns: Tuple[Callable, ...], probs: Tuple[float, ...]
) -> Array:
    """
    Applies a random augmentation to each image in a batch using jax.vmap.
    """
    rngs_per_image = jax.random.split(rng, num=images.shape[0])
    jnp_probs = jnp.array(probs)
    vmapped_augment = jax.vmap(
        apply_random_augmentation_to_one_image, in_axes=(0, 0, None, None)
    )
    return vmapped_augment(images, rngs_per_image, aug_fns, jnp_probs)

# --- Display Utility ---
def display_cifar_images(images, num_samples=16, rows=4, cols=4, title:str="Images"):
    images_np = np.array(images)
    # Transpose from (N, C, H, W) to (N, H, W, C) for matplotlib
    images_np = np.transpose(images_np, (0, 2, 3, 1))
    # Denormalize from [-1, 1] to [0, 1]
    images_np = np.clip((images_np + 1) / 2.0, 0, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(4, 4))
    fig.frameon = False
    fig.suptitle(title, y=0.95, fontsize=14)
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            ax.imshow(images_np[i])
        ax.axis('off')
    plt.tight_layout(pad=.1, rect=(0, 0, 1, 0.93))
    plt.show()


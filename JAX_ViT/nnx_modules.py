import jax
import jax.numpy as jnp
from typing import TypeAlias
from flax import nnx
from dataclasses import dataclass

Array: TypeAlias = jax.Array

class Attention(nnx.Module):
    """Standard Multi-Head Attention using Flax's efficient `dot_product_attention` implementation."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Fused QKV projection
        self.qkv = nnx.Linear(dim, dim * 3, use_bias=qkv_bias, rngs=rngs)
        # Output projection
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        B, N, C = x.shape
        # Project to Q, K, V and reshape for multi-head attention
        # (B, N, 3 * D) -> (B, N, 3, H, D_h) -> (3, B, N, H, D_h)
        qkv = jnp.moveaxis(self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim), 2, 0) # or .transpose(2, 0, 1, 3, 4)
        q_k_v = jnp.split(qkv, 3, 0) # Split along the first dimension
        q, k, v = [i.squeeze(0) for i in q_k_v]

        attn_output = nnx.dot_product_attention(q, k, v, broadcast_dropout=False)

        # Reshape and project back to original dimension
        # (B, H, N, D_h) -> (B, N, H, D_h) -> (B, N, D)
        out = attn_output.swapaxes(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out
    
class SiLU(nnx.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Array) -> Array:
        return nnx.silu(x)

class FeedForward(nnx.Module):
    """Standard Feed-Forward Network found in Transformers."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, rngs: nnx.Rngs = nnx.Rngs(0)):
        super().__init__()
        self.net = nnx.Sequential(
            nnx.Linear(dim, hidden_dim, rngs=rngs),
            SiLU(), 
            nnx.Dropout(dropout, rngs=rngs),
            nnx.Linear(hidden_dim, dim, rngs=rngs),
            nnx.Dropout(dropout, rngs=rngs)
        )
    def __call__(self, x: Array) -> Array:
        return self.net(x)

class GLU(nnx.Module):
    """Gated Linear Unit variant: Sigmoid(W_gate*x) * (W_transform*x)."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float, bias: bool = False, *, rngs: nnx.Rngs):
        self.fc_gate_value = nnx.Linear(in_features, 2 * hidden_dim, use_bias=bias, rngs=rngs)
        self.fc_out = nnx.Linear(hidden_dim, out_features, use_bias=bias, rngs=rngs)
        self.dropout = nnx.Dropout(rate=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        gate_val_proj = self.fc_gate_value(x)
        gated_values, gate_input = jnp.split(gate_val_proj, 2, axis=-1)

        gated_activation = nnx.sigmoid(gated_values) * gate_input
        gated_activation = self.dropout(gated_activation)
        
        out = self.fc_out(gated_activation)
        return out

class TransformerBlock(nnx.Module):
    """A single Transformer Block."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float, qkv_bias: bool = False, *, rngs: nnx.Rngs):
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = Attention(dim, num_heads, qkv_bias, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.ffn = GLU(in_features=dim, hidden_dim=mlp_hidden_dim, out_features=dim, dropout=dropout, rngs=rngs)

    def __call__(self, x: Array) -> Array:
        # Pre-normalization variant
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


image_shape: TypeAlias = tuple[int, int, int]

@dataclass
class ViTConfig:
    img_shape: image_shape = (32, 32, 3)
    patch_size: int = 1
    num_classes: int = 10
    dim: int = 32
    depth: int = 6
    heads: int = 8
    mlp_ratio: float = 4.0
    channels: int = 3
    dropout: float = 0.1
    rngs: nnx.Rngs = nnx.Rngs(0)


class ConvPatchify(nnx.Module):
    def __init__(self, channels:int, out_dim:int, rngs: nnx.Rngs = nnx.Rngs(0)) -> None:
        super().__init__()
        self.conv1 = nnx.Conv(channels, 32, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.pool1 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(1, 1), padding=((1, 1), (1, 1))) # type: ignore
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.pool2 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(1, 1), padding=((1, 1), (1, 1))) # type: ignore
        self.conv3 = nnx.Conv(64, 32, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.pool3 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.conv4 = nnx.Conv(32, out_dim, kernel_size=(3, 3), padding=1, rngs=rngs)
        self.pool4 = lambda x: nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        self.relu = nnx.relu

    def __call__(self, x: Array) -> Array:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.relu(self.conv4(x))
        x = self.pool4(x)
        return x


class ViT(nnx.Module):
    """
    Vision Transformer with a CNN-based patchifier (Hybrid ViT: ConViT).
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        rngs = config.rngs
        dim = config.dim

        # 1. CNN Feature Extractor (Patchifier)
        self.conv_patchifier = ConvPatchify(config.channels, config.dim, rngs=rngs)
        img_size = jnp.array(config.img_shape[:-1]).prod().item()
        feature_map_size = img_size // 4**2 # two maxpools with stride 2
        assert feature_map_size == 8**2, f"Feature map size calculation is wrong. Expected 8 instead got {feature_map_size}"
        
        num_patches = (feature_map_size // config.patch_size)
        
        # This is an alternative, more direct way to patchify, but we use the CNN above (for the sake of using CNNs)
        # self.patch_embedding = nnx.conv(3, dim, kernel_size=patch_size, strides=patch_size)
        
        # 2. Transformer specific parameters
        self.pos_embedding = nnx.Param(jax.random.normal(rngs(), (1, num_patches + 1, dim))*.01)
        self.cls_token = nnx.Param(jax.random.normal(rngs(), (1, 1, dim))*.01)
        self.dropout = nnx.Dropout(config.dropout, rngs=rngs)
        
        # 3. Stack of Transformer Blocks
        self.transformer_blocks = [
            TransformerBlock(dim=dim, num_heads=config.heads, mlp_ratio=config.mlp_ratio, dropout=config.dropout, rngs=rngs)
            for _ in range(config.depth)
        ]
        
        # 4. Classifier Head
        self.classifier_head = nnx.Sequential(
            nnx.LayerNorm(dim, rngs=rngs),
            nnx.Linear(dim, config.num_classes, kernel_init=nnx.initializers.uniform(0.001), rngs=rngs)
        )

    def __call__(self, img: Array) -> Array:
        B = img.shape[0]

        # 1. Create patch embeddings using the CNN
        x = self.conv_patchifier(img) # Shape: (B, H', W', dim) e.g., (B, 8, 8, dim)
        
        # Flatten for transformer: (B, H' * W', dim)
        x = x.reshape(B, -1, x.shape[-1])
        N = x.shape[1] # Number of patches

        # 2. Prepend CLS token
        cls_tokens = jnp.broadcast_to(self.cls_token.value, (B, 1, x.shape[-1]))
        x = jnp.concatenate((cls_tokens, x), axis=1) # Shape: (B, N + 1, dim)
        
        # 3. Add positional embedding
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)
        
        # 4. Pass through transformer blocks
        transformer_in = x
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)
            if i % 2 == 0:
                residual = x
            else:
                x += residual
        x += transformer_in
        # 5. Get the CLS token output and classify
        cls_output = x[:, 0]
        return self.classifier_head(cls_output)
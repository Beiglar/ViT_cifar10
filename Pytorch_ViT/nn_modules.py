import torch
from torch import nn
from torch.nn import functional as F
from typing import TypeAlias
import numpy as np

# Model and Modual Definition

class Attention(nn.Module):
    """Standard Multi-Head Attention using PyTorch's efficient implementation."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Output projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project to Q, K, V and reshape for multi-head attention
        # (B, N, 3 * D) -> (B, N, 3, H, D_h) -> (3, B, H, N, D_h)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # Unbind along the first dimension

        # Use PyTorch 2.0's efficient attention
        # For older PyTorch, you'd do: (q @ k.transpose(-2, -1)) * self.scale
        # followed by softmax and matmul with v.
        attn_output = F.scaled_dot_product_attention(q, k, v)

        # Reshape and project back to original dimension
        # (B, H, N, D_h) -> (B, N, H, D_h) -> (B, N, D)
        out = attn_output.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    """Standard Feed-Forward Network found in Transformers."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(), # GELU is standard in ViT
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GLU(nn.Module):
    """Gated Linear Unit variant: Sigmoid(W_gate*x) * (W_transform*x)."""
    def __init__(self, in_features: int, hidden_dim: int, out_features: int, dropout: float = 0.5, bias: bool = False):
        super().__init__()
        # Linear layer for gate and value, then split. hidden_features is for each branch before *
        self.fc_gate_value = nn.Linear(in_features, 2 * hidden_dim, bias=bias)
        self.fc_out = nn.Linear(hidden_dim, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_val_proj: torch.Tensor = self.fc_gate_value(x)
        # Split into two parts: one for the gate, one for the value
        gated_values, gate_input = torch.split(gate_val_proj, gate_val_proj.shape[-1] // 2, dim=-1)

        gated_activation = self.dropout(self.sigmoid(gated_values) * gate_input) # Element-wise product
        out = self.fc_out(gated_activation)
        return out

class TransformerBlock(nn.Module):
    """A single Transformer Block."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = GLU(in_features=dim, hidden_dim=int(dim * mlp_ratio), out_features=dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-normalization is a common and stable variant
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

image_shape: TypeAlias = tuple[int, int, int]

class ViT(nn.Module):
    """
    Vision Transformer with a CNN-based patchifier (Hybrid ViT).
    """
    def __init__(
            self,
            *,
            img_shape: image_shape,
            patch_size: int = 4, # The size of patches to be extracted from the feature map
            num_classes: int,
            dim: int,
            depth: int,
            heads: int,
            mlp_ratio: float = 4.0,
            channels: int = 3,
            dropout: float = 0.1
            ):
        super().__init__()
        
        # 1. CNN Feature Extractor (Patchifier)
        # This block correctly downsamples the image.
        # Input: 32x32x3 -> Conv -> 32x32x16 -> Pool -> 16x16x16 -> Conv -> 16x16x32 -> Pool -> 8x8x32
        self.conv_patchifier = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1), # 32 -> 32
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1), # 32 -> 32
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1), # 32 -> 16
            nn.Conv2d(32, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16 -> 8
        )
        img_size = np.array(img_shape[:-1]).prod()
        # After conv_patchifier, feature map is (dim, 8, 8)
        feature_map_size = img_size // 4**2 # two maxpools with stride 2
        assert feature_map_size == 64, f"Feature map size calculation is wrong. Expected 8 instead got {feature_map_size}"
        
        num_patches = (feature_map_size // patch_size) ** 2
        
        # This is an alternative, more direct way to patchify, but we use the CNN above
        # self.patch_embedding = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # 2. Transformer specific parameters
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)
        
        # 3. Stack of Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(depth)
        ])
        
        # 4. Classifier Head
        self.to_latent = nn.Identity()
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # 1. Create patch embeddings using the CNN
        x = self.conv_patchifier(img) # Shape: (B, dim, H', W') -> (B, 64, 8, 8)
        # Flatten and transpose for transformer input
        x = x.flatten(2).transpose(1, 2) # Shape: (B, num_patches, dim) -> (B, 64, 64)
        B, N, _ = x.shape

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (B, num_patches + 1, dim)
        
        # 3. Add positional embedding
        x += self.pos_embedding[:, :(N + 1)]
        x = self.dropout(x)
        
        # 4. Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
            
        # 5. Get the CLS token output and classify
        cls_output = self.to_latent(x[:, 0])
        return self.classifier_head(cls_output)

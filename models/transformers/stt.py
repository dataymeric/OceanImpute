import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..layers import MLP
from ..embeddings import (
    PatchEmbed,
    PatchEmbed3d,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention layer."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.size()
        qkv = self.qkv(x)  # b, n, d * 3

        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask):
        # query/value: spatiotemporal tokens; key: mask tokens
        b, n, d = x.shape

        q = self.q(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv(mask).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        x = attn @ v
        x = x.transpose(1, 2).reshape(b, n, d)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """
    A transformer block with or without adaptive layer norm zero (adaLN-Zero)
    conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_modulation=False):
        super().__init__()
        self.use_modulation = use_modulation
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))
        if use_modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True),
            )

    def forward(self, x, mask):
        if self.use_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mask).chunk(6, dim=2)
            )
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionTransformerBlock(nn.Module):
    """A transformer block with cross-attention conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), mask)
        x = x + self.mlp(self.norm3(x))
        return x


class FinalLayer(nn.Module):
    """The final layer of the Transformer."""

    def __init__(self, hidden_size, num_patches, out_channels, use_modulation=False):
        super().__init__()
        self.use_modulation = use_modulation
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patches * out_channels, bias=True)
        if use_modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * hidden_size, bias=True),
            )

    def forward(self, x, mask):
        if self.use_modulation:
            shift, scale = self.adaLN_modulation(mask).chunk(2, dim=2)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class Transformer2d(nn.Module):
    """Simple spatiotemporal vision transformer."""

    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(2)])
        self.num_patches = num_patches
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, np.prod(patch_size) * self.out_channels),
        )

        self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())

    def get_spatial_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[1] // self.patch_size[1],
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def unpatchify(self, x):
        """For use with PatchEmbed.

        (B, N, C) -> (B, T, H, W)
        """
        c = self.out_channels  # time dimension
        h, w = [self.input_size[i] // self.patch_size[i] for i in range(2)]
        ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], h, w, ph, pw, c))
        x = rearrange(x, "b h w ph pw c -> b c h ph w pw")
        x = x.reshape(shape=(x.shape[0], c, h * ph, w * pw))
        return x

    def forward(self, x):
        """Forward pass of the transformer.

        Args:
            x (torch.Tensor): inputs of shape (B, T, H, W)

        Returns:
            torch.Tensor: outputs of shape (B, out_channels, H, W)
        """
        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = x + self.pos_embed_spatial

        # transformer blocks
        for block in self.blocks:
            x = block(x)  # (B, N, D)

        # final process
        x = self.final_layer(x)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)
        return x


class Transformer3d(nn.Module):
    def __init__(
        self,
        input_size,
        in_channels,
        out_channels,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio=4.0,
        use_modulation=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.use_modulation = use_modulation

        self.x_embedder = PatchEmbed3d(patch_size, in_channels, hidden_size)
        if use_modulation:
            self.mask_embedder = PatchEmbed3d(patch_size, in_channels, hidden_size)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    use_modulation=use_modulation,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size,
            np.prod(patch_size),
            out_channels,
            use_modulation=use_modulation,
        )

        self.register_buffer("pos_embed_spatial", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.initialize_weights()

    def get_spatial_pos_embed(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[1] // self.patch_size[1],
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def add_temporal_pos_embed(self, x):
        """Add temporal positional embedding to the input tensor."""
        x = rearrange(
            x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial
        )
        x = x + self.pos_embed_spatial
        x = rearrange(x, "b t s d -> b s t d")
        x = x + self.pos_embed_temporal
        x = rearrange(x, "b s t d -> b (t s) d")
        return x

    def unpatchify(self, x):
        """For use with PatchEmbed3d.

        (B, N, C) -> (B, C, T, H, W)
        """
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "b t h w pt ph pw c -> b c t pt h ph w pw")
        x = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return x

    def forward(self, x, mask):
        """Forward pass of the transformer.

        Args:
            x (torch.Tensor): inputs of shape (B, in_channels, T, H, W)
            mask (torch.Tensor): mask of shape (B, in_channels, T, H, W)

        Returns:
            torch.Tensor: outputs of shape (B, out_channels, T, H, W)
        """
        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = self.add_temporal_pos_embed(x)

        if self.use_modulation:
            mask = self.mask_embedder(mask)  # (B, N, D)
            mask = self.add_temporal_pos_embed(mask)
        else:
            mask = None

        # transformer blocks
        for block in self.blocks:
            x = block(x, mask)  # (B, N, D)

        # final process
        x = self.final_layer(x, mask)  # (B, N, num_patches * out_channels)
        x = self.unpatchify(x)  # (B, out_channels, T, H, W)
        return x

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad_:
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch embedding like nn.Linear (instead of nn.Conv3d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.use_modulation:
            w = self.mask_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.mask_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in transformer blocks:
        for block in self.blocks:
            if hasattr(block, "adaLN_modulation"):
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)
        if hasattr(self.final_layer, "adaLN_modulation"):
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

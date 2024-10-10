import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    """Spatiotemporal to Patch Embedding, where the temporal dimension is treated as
    the input channels dimension.

    Args:
        patch_size (tuple, optional): Patch token size. Default: (4, 4).
        in_chans (int, optional): Number of input channels. Default: 1.
        embed_dim (int, optional): Size of the embedding. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        flatten (bool, optional): If `True`, flatten the projection. Default: True.

    Shape:
        - Input: (B, T, H, W)
        - Output: (B, N, C) where N = H // patch_size[0] * W // patch_size[1] and
        C = embed_dim.
    """

    def __init__(
        self,
        patch_size=(4, 4),
        in_chans=1,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        b, t, h, w = x.size()

        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbed2d(nn.Module):
    """Spatiotemporal to Patch Embedding, where the temporal dimension is concatenated
    to the spatial dimensions.

    This embedding is inspired by the "Uniform frame sampling" from
    https://arxiv.org/abs/2103.15691.

    Args:
        patch_size (tuple, optional): Patch token size. Default: (4, 4).
        in_chans (int, optional): Number of input channels. Default: 1.
        embed_dim (int, optional): Size of the embedding. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        flatten (bool, optional): If `True`, flatten the projection. Default: True.

    Shape:
        - Input: (B, C, T, H, W)
        - Output: (B, N, C) where N = T * H // patch_size[1] * W // patch_size[2] and
        C = embed_dim.
    """

    def __init__(
        self,
        patch_size=(4, 4),
        in_chans=1,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")

        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
            x = rearrange(x, "(b t) n c -> b (t n) c", b=b, t=t)
        else:
            x = rearrange(x, "(b t) n c -> b c t h w", b=b, t=t)
        return x


class PatchEmbed3d(nn.Module):
    """Spatiotemporal to Patch Embedding.

    This embedding is inspired by the "Tubelet embedding" from
    https://arxiv.org/abs/2103.15691.

    Args:
        patch_size (tuple, optional): Patch token size. Default: (2, 4, 4).
        in_chans (int, optional): Number of input channels. Default: 1.
        embed_dim (int, optional): Size of the embedding. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        flatten (bool, optional): If `True`, flatten the projection. Default: True.

    Shape:
        - Input: (B, C, T, H, W)
        - Output: (B, N, C)
        where N = T // patch_size[0] * H // patch_size[1] * W // patch_size[2] and
        C = embed_dim.
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=1,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        b, c, t, h, w = x.size()
        # Padding
        if w % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
        if h % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
        if t % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - t % self.patch_size[0]))

        x = self.proj(x)  # (B, C, T, Wh, Ww)
        if self.norm is not None:
            t, wh, ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, t, wh, ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, C, T, H, W) -> (B, N, C)
        return x


"""
# Sine/Cosine Positional Embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
"""


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim]
    (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

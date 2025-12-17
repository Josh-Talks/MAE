import collections.abc
import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Union, Callable, Optional, List, Literal, Tuple


from .utils import ensure_tuple
from .weight_init import trunc_normal_


class PatchEmbed(nn.Module):
    """2D/3D Patch Embedding that converts image into patch tokens"""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        in_channels: int = 3,
        embed_dim: int = 768,
        spatial_dims: int = 2,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
    ):
        super().__init__()
        img_size = ensure_tuple(img_size, dim=spatial_dims)
        patch_size = ensure_tuple(patch_size, dim=spatial_dims)

        for m, p in zip(img_size, patch_size):
            if m < p:
                raise ValueError("patch_size should be smaller than img_size.")

        self.num_patches = np.prod(
            [im_i // p_j for im_i, p_j in zip(img_size, patch_size)]
        )

        if spatial_dims == 2:
            assert len(patch_size) == 2
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        elif spatial_dims == 3:
            assert len(patch_size) == 3
            self.proj = nn.Conv3d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            )
        else:
            raise ValueError("spatial_dims must be 2 or 3")
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B, C, spatial (e.g 2D=(H, W) or 3D=(D, H, W))
        x = x.flatten(2).transpose(-1, -2)  # B, N, C
        x = self.norm(x)
        return x


class PositionEmbed(nn.Module):
    """Positional Encoding using learnable parameters"""

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        spatial_dims: int,
        img_size: Optional[Union[Sequence[int], int]] = None,
        patch_size: Optional[Union[Sequence[int], int]] = None,
        pos_embed_type: Literal["none", "learnable", "sincos"] = "sincos",
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if pos_embed_type == "none":
            self.pos_embed.requires_grad = False
        elif pos_embed_type == "learnable":
            _ = trunc_normal_(self.pos_embed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        elif pos_embed_type == "sincos":
            assert (
                img_size is not None and patch_size is not None
            ), "img_size and patch_size must be provided for sincos positional embedding"
            img_size = ensure_tuple(img_size, dim=spatial_dims)
            patch_size = ensure_tuple(patch_size, dim=spatial_dims)
            grid_size: List[int] = []
            for in_size, pa_size in zip(img_size, patch_size):
                grid_size.append(in_size // pa_size)

            self.pos_embed = build_sincos_position_embedding(
                grid_size, embed_dim, spatial_dims
            )
        else:
            raise ValueError(f"pos_embed_type {self.pos_embed_type} not supported.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_embed
        return x


# From PyTorch internals
def _ntuple(n: int):

    def parse(x: List[int]) -> Tuple[int, ...]:
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        else:
            raise TypeError("Input must be an iterable")

    return parse


def build_sincos_position_embedding(
    grid_size: List[int],
    embed_dim: int,
    spatial_dims: int = 3,
    temperature: float = 10000.0,
) -> torch.nn.Parameter:
    """
    Builds a sin-cos position embedding based on the given grid size, embed dimension, spatial dimensions, and temperature.
    Reference: https://github.com/cvlab-stonybrook/SelfMedMAE/blob/68d191dfcc1c7d0145db93a6a570362de29e3b30/lib/models/mae3d.py

    Args:
        grid_size (List[int]): The size of the grid in each spatial dimension.
        embed_dim (int): The dimension of the embedding.
        spatial_dims (int): The number of spatial dimensions (2 for 2D, 3 for 3D).
        temperature (float): The temperature for the sin-cos position embedding.

    Returns:
        pos_embed (nn.Parameter): The sin-cos position embedding as a fixed parameter.
    """

    if spatial_dims == 2:
        to_2tuple = _ntuple(2)
        grid_size_t = to_2tuple(grid_size)
        h, w = grid_size_t
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)

        grid_h, grid_w = torch.meshgrid(grid_h, grid_w)

        if embed_dim % 4 != 0:
            raise AssertionError(
                "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
            )

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]
    elif spatial_dims == 3:
        to_3tuple = _ntuple(3)
        grid_size_t = to_3tuple(grid_size)
        h, w, d = grid_size_t
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_d = torch.arange(d, dtype=torch.float32)

        grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)

        if embed_dim % 6 != 0:
            raise AssertionError(
                "Embed dimension must be divisible by 6 for 3D sin-cos position embedding"
            )

        pos_dim = embed_dim // 6
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_d = torch.einsum("m,d->md", [grid_d.flatten(), omega])
        pos_emb = torch.cat(
            [
                torch.sin(out_w),
                torch.cos(out_w),
                torch.sin(out_h),
                torch.cos(out_h),
                torch.sin(out_d),
                torch.cos(out_d),
            ],
            dim=1,
        )[None, :, :]
    else:
        raise NotImplementedError(
            "Spatial Dimension Size {spatial_dims} Not Implemented!"
        )

    pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False

    return pos_embed

import torch
import torch.nn as nn
from typing import Sequence, Union

from MAE.embedding import PatchEmbed
from MAE.weight_init import trunc_normal_


class MaskedAutoencoder(nn.Module):
    """MaskedAutoencoder with ViT backbone"""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        mask_ratio: float = 0.75,
        spatial_dims: int = 2,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            spatial_dims=spatial_dims,
        )
        num_patches = self.patch_embed.num_patches

        self.mask_ratio = mask_ratio

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

    def initialise_weights(self):
        w = self.patch_embed.proj.weight.data
        _ = torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize patch_embedding like nn.Linear (instead of nn.Conv2d)
        trunc_normal_(self.mask_tokens, mean=0.0, std=0.02, a=-2.0, b=2.0)
        trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

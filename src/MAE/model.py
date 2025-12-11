import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Union, Literal, Tuple

from MAE.embedding import PatchEmbed, PositionEmbed
from MAE.weight_init import trunc_normal_
from MAE.ViT import TransformerBlock


class MaskedAutoencoder(nn.Module):
    """MaskedAutoencoder with ViT backbone"""

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        patch_size: Union[Sequence[int], int],
        in_channels: int,
        embed_dim: int,
        num_classes: int,
        depth: int,
        num_heads: int,
        decoder_embed_dim: int,
        decoder_depth: int,
        decoder_num_heads: int,
        pos_embed_type: Literal["none", "learnable", "sincos"] = "sincos",
        qkv_bias: bool = True,
        dropout_rate: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        mask_ratio: float = 0.75,
        mlp_ratio: float = 4.0,
        spatial_dims: int = 2,
    ):
        super().__init__()

        self.num_classes = num_classes

        # --------------------------------------------------------------------------
        # MAE encoder

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            spatial_dims=spatial_dims,
        )
        num_patches = int(self.patch_embed.num_patches)

        self.mask_ratio = mask_ratio

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = PositionEmbed(
            num_patches=num_patches,
            embed_dim=embed_dim,
            spatial_dims=spatial_dims,
            img_size=img_size,
            patch_size=patch_size,
            pos_embed_type=pos_embed_type,
        )

        blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias,
                dropout_rate,
                attn_drop,
                drop_path,
            )
            for _ in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks, nn.LayerNorm(embed_dim))

        # --------------------------------------------------------------------------
        # MAE decoder

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = PositionEmbed(
            num_patches=num_patches,
            embed_dim=decoder_embed_dim,
            spatial_dims=spatial_dims,
            img_size=img_size,
            patch_size=patch_size,
            pos_embed_type=pos_embed_type,
        )

        decoder_blocks = [
            TransformerBlock(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias,
                dropout_rate,
                attn_drop,
                drop_path,
            )
            for _ in range(decoder_depth)
        ]

        self.decoder_blocks = nn.Sequential(
            *decoder_blocks, nn.LayerNorm(decoder_embed_dim)
        )

        self.decoder_pred = nn.Linear(
            decoder_embed_dim, int(np.prod(patch_size)) * in_channels, bias=True
        )

        self.initialize_weights()

    def initialize_weights(self):
        w = self.patch_embed.proj.weight.data
        _ = torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize patch_embedding like nn.Linear (instead of nn.Conv2d)
        _ = trunc_normal_(self.mask_token, mean=0.0, std=0.02, a=-2.0, b=2.0)
        _ = trunc_normal_(self.cls_token, mean=0.0, std=0.02, a=-2.0, b=2.0)

        _ = self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):

        if isinstance(m, nn.Linear):
            _ = nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:  # pyright: ignore
                _ = nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            _ = nn.init.zeros_(m.bias)
            _ = nn.init.ones_(m.weight)

    def random_masking(self, x: torch.Tensor):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [B, L, D], sequence
        """
        B, L, D = x.shape  # batch, length, dim
        n_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :n_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :n_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # --------------------------------------------------------------------------
        # MAE encoder

        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = self.pos_embed(x)

        x, mask, ids_restore = self.random_masking(x)

        # combine cls tokens with the patch tokens
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        x = self.blocks(x)

        # --------------------------------------------------------------------------
        # MAE decoder
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle

        # add pos embed
        x_ = self.decoder_pos_embed(x_)

        # append cls token
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        x = self.decoder_blocks(x)
        x = self.decoder_pred(x)

        x = x[:, 1:, :]  # remove cls token

        return x, mask

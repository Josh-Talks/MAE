import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Union, Literal, Tuple

from MAE.embedding import PatchEmbed, PositionEmbed
from MAE.utils import ensure_tuple
from MAE.ViT import TransformerBlock
from MAE.weight_init import trunc_normal_


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

        assert (
            0 <= dropout_rate <= 1
        ), f"dropout_rate should be between 0 and 1, got {dropout_rate}."
        assert (
            0 <= attn_drop <= 1
        ), f"attn_drop should be between 0 and 1, got {attn_drop}."
        assert (
            0 <= drop_path <= 1
        ), f"drop_path should be between 0 and 1, got {drop_path}."
        assert embed_dim % num_heads == 0, "embed_dim should be divisible by num_heads"
        assert (
            decoder_embed_dim % decoder_num_heads == 0
        ), "decoder_embed_dim should be divisible by decoder_num_heads"

        self.num_classes = num_classes
        self.img_size = ensure_tuple(img_size, dim=spatial_dims)
        self.patch_size = ensure_tuple(patch_size, dim=spatial_dims)
        self.in_channels = in_channels
        self.spatial_dims = spatial_dims

        # --------------------------------------------------------------------------
        # MAE encoder

        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
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
            img_size=self.img_size,
            patch_size=self.patch_size,
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
            img_size=self.img_size,
            patch_size=self.patch_size,
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
        self.output_dim = int(np.prod(self.patch_size)) * in_channels
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.output_dim, bias=True)

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

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reshape patches back to original image dimensions without modifying values.
        Assumes patches are in row-major order (for 2D) or depth-row-major order (for 3D).

        x: (N, L, prod(patch_size) * in_channels)
        Returns: (N, in_channels, *img_size)
        """
        if self.spatial_dims == 2:
            # x shape: (N, h*w, p*p*c)
            p_h, p_w = self.patch_size[0], self.patch_size[1]
            h = self.img_size[0] // p_h  # number of patches in height
            w = self.img_size[1] // p_w  # number of patches in width

            # Reshape to (N, h, w, p_h, p_w, c)
            img = x.view(-1, h, w, p_h, p_w, self.in_channels)

            # Rearrange to (N, c, h, p_h, w, p_w) then (N, c, h*p_h, w*p_w)
            img = img.permute(0, 5, 1, 3, 2, 4).contiguous()
            img = img.view(-1, self.in_channels, self.img_size[0], self.img_size[1])

        elif self.spatial_dims == 3:
            # x shape: (N, d*h*w, p_d*p_h*p_w*c)
            p_d, p_h, p_w = self.patch_size[0], self.patch_size[1], self.patch_size[2]
            d = self.img_size[0] // p_d  # number of patches in depth
            h = self.img_size[1] // p_h  # number of patches in height
            w = self.img_size[2] // p_w  # number of patches in width

            # Reshape to (N, d, h, w, p_d, p_h, p_w, c)
            img = x.view(-1, d, h, w, p_d, p_h, p_w, self.in_channels)

            # Rearrange to (N, c, d, p_d, h, p_h, w, p_w) then (N, c, d*p_d, h*p_h, w*p_w)
            img = img.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            img = img.view(
                -1,
                self.in_channels,
                self.img_size[0],
                self.img_size[1],
                self.img_size[2],
            )
        else:
            raise ValueError("spatial_dims must be 2 or 3")

        return img

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Convert images into patches.
        Assumes images are in (N, in_channels, *img_size) format.

        imgs: (N, in_channels, *img_size)
        Returns: (N, L, prod(patch_size) * in_channels)
        """
        N = imgs.shape[0]
        if self.spatial_dims == 2:
            p_h, p_w = self.patch_size[0], self.patch_size[1]
            h, w = self.img_size[0], self.img_size[1]
            assert (
                h % p_h == 0 and w % p_w == 0
            ), "Image dimensions must be divisible by patch size."

            # Reshape to (N, c, h//p_h, p_h, w//p_w, p_w)
            x = imgs.view(N, self.in_channels, h // p_h, p_h, w // p_w, p_w)

            # Rearrange to (N, h//p_h, w//p_w, p_h, p_w, c) then (N, L, p_h*p_w*c)
            x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
            x = x.view(N, -1, p_h * p_w * self.in_channels)

        elif self.spatial_dims == 3:
            p_d, p_h, p_w = self.patch_size[0], self.patch_size[1], self.patch_size[2]
            d, h, w = self.img_size[0], self.img_size[1], self.img_size[2]
            assert (
                d % p_d == 0 and h % p_h == 0 and w % p_w == 0
            ), "Image dimensions must be divisible by patch size."

            # Reshape to (N, c, d//p_d, p_d, h//p_h, p_h, w//p_w, p_w)
            x = imgs.view(
                N, self.in_channels, d // p_d, p_d, h // p_h, p_h, w // p_w, p_w
            )

            # Rearrange to (N, d//p_d, h//p_h, w//p_w, p_d, p_h, p_w, c) then (N, L, p_d*p_h*p_w*c)
            x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
            x = x.view(N, -1, p_d * p_h * p_w * self.in_channels)
        else:
            raise ValueError("spatial_dims must be 2 or 3")
        return x

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
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
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

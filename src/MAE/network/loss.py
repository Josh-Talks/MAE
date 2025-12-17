import torch
import torch.nn as nn

from .utils import patchify
from .model import MaskedAutoencoder


def mask_patched(input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask patches corresponding to mask argument

    Args:
        input (torch.Tensor): patched input of shape [B, N, D]
        mask (torch.Tensor): mask tensor of shape [B, N]

    Returns:
        torch.Tensor: masked input of shape [B, M, D] where M is the number of patches where mask == 1
    """
    mask_expanded = mask.unsqueeze(-1)  # Shape: [B, N, 1]
    masked_input = input[mask_expanded.expand_as(input) == 1].reshape(
        input.shape[0], -1, input.shape[-1]
    )
    return masked_input


class MSELossPatched(nn.Module):
    def __init__(self):
        super(MSELossPatched, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        model: MaskedAutoencoder,
    ) -> torch.Tensor:
        target_patched = patchify(
            target,
            model.patch_size,
            model.img_size,
            model.in_channels,
            model.spatial_dims,
        )
        target_masked = mask_patched(target_patched, mask)
        pred_masked = mask_patched(pred, mask)

        return self.loss_fn(pred_masked, target_masked)

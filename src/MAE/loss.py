import torch
import torch.nn as nn

from MAE.dataclass import DataDimensions
from MAE.utils import patchify


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


class MSELoss_patched(nn.Module):
    def __init__(self):
        super(MSELoss_patched, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        data_dims: DataDimensions,
    ) -> torch.Tensor:
        target_patched = patchify(
            target,
            data_dims.patch_size,
            data_dims.img_size,
            data_dims.in_channels,
            data_dims.spatial_dims,
        )
        target_masked = mask_patched(target_patched, mask)
        pred_masked = mask_patched(pred, mask)

        return self.loss_fn(pred_masked, target_masked)

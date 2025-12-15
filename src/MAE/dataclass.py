from pydantic import BaseModel
from typing import Tuple


class DataDimensions(BaseModel):
    spatial_dims: int
    in_channels: int
    img_size: Tuple[int, ...]
    patch_size: Tuple[int, ...]

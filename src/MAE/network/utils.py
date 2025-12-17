import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import shutil
import torch
import torch.nn as nn
from typing import Any, Dict, Sequence, Tuple, TypeGuard, Union, List, Optional


def is_ndarray(v: Any) -> TypeGuard[NDArray[Any]]:
    return isinstance(v, np.ndarray)


def ensure_tuple(x: Union[Sequence[int], int], dim: int) -> Tuple[int, ...]:
    """Ensure that the input is a tuple of given dimension.

    If the input is an integer, it will be converted to a tuple with the integer
    repeated `dim` times. If the input is already a sequence, it will be converted
    to a tuple.

    Args:
        x (Union[Sequence[int], int]): Input value to ensure as tuple.
        dim (int): Desired dimension of the output tuple.

    Returns:
        Tuple[int, ...]: Tuple of integers with length `dim`.
    """
    if isinstance(x, int):
        return (x,) * dim
    elif isinstance(x, Sequence):
        if len(x) != dim:
            raise ValueError(f"Input sequence must have length {dim}, but got {len(x)}")
        return tuple(x)
    else:
        raise TypeError("Input must be an integer or a sequence of integers.")


def unpatchify(
    x: torch.Tensor,
    patch_size: Tuple[int, ...],
    img_size: Tuple[int, ...],
    in_channels: int,
    spatial_dims: int,
) -> torch.Tensor:
    """
    Reshape patches back to original image dimensions without modifying values.
    Assumes patches are in row-major order (for 2D) or depth-row-major order (for 3D).

    x: (N, L, prod(patch_size) * in_channels)
    Returns: (N, in_channels, *img_size)
    """
    if spatial_dims == 2:
        # x shape: (N, h*w, p*p*c)
        p_h, p_w = patch_size[0], patch_size[1]
        h = img_size[0] // p_h  # number of patches in height
        w = img_size[1] // p_w  # number of patches in width

        # Reshape to (N, h, w, p_h, p_w, c)
        img = x.view(-1, h, w, p_h, p_w, in_channels)

        # Rearrange to (N, c, h, p_h, w, p_w) then (N, c, h*p_h, w*p_w)
        img = img.permute(0, 5, 1, 3, 2, 4).contiguous()
        img = img.view(-1, in_channels, img_size[0], img_size[1])
    elif spatial_dims == 3:
        # x shape: (N, d*h*w, p_d*p_h*p_w*c)
        p_d, p_h, p_w = patch_size[0], patch_size[1], patch_size[2]
        d = img_size[0] // p_d  # number of patches in depth
        h = img_size[1] // p_h  # number of patches in height
        w = img_size[2] // p_w  # number of patches in width

        # Reshape to (N, d, h, w, p_d, p_h, p_w, c)
        img = x.view(-1, d, h, w, p_d, p_h, p_w, in_channels)
        # Rearrange to (N, c, d, p_d, h, p_h, w, p_w) then (N, c, d*p_d, h*p_h, w*p_w)
        img = img.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        img = img.view(
            -1,
            in_channels,
            img_size[0],
            img_size[1],
            img_size[2],
        )
    else:
        raise ValueError("spatial_dims must be 2 or 3")

    return img


def patchify(
    imgs: torch.Tensor,
    patch_size: Tuple[int, ...],
    img_size: Tuple[int, ...],
    in_channels: int,
    spatial_dims: int,
) -> torch.Tensor:
    """
    Convert images into patches.
    Assumes images are in (N, in_channels, *img_size) format.

    imgs: (N, in_channels, *img_size)
    Returns: (N, L, prod(patch_size) * in_channels)
    """
    assert (
        len(patch_size) == spatial_dims
    ), f"patch_size must have length {spatial_dims} for {spatial_dims}D data"
    assert (
        len(img_size) == spatial_dims
    ), f"img_size must have length {spatial_dims} for {spatial_dims}D data"

    N = imgs.shape[0]
    if spatial_dims == 2:
        p_h, p_w = patch_size[0], patch_size[1]
        h, w = img_size[0], img_size[1]
        assert (
            h % p_h == 0 and w % p_w == 0
        ), "Image dimensions must be divisible by patch size."

        # Reshape to (N, c, h//p_h, p_h, w//p_w, p_w)
        x = imgs.view(N, in_channels, h // p_h, p_h, w // p_w, p_w)

        # Rearrange to (N, h//p_h, w//p_w, p_h, p_w, c) then (N, L, p_h*p_w*c)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(N, -1, p_h * p_w * in_channels)

    elif spatial_dims == 3:
        p_d, p_h, p_w = patch_size[0], patch_size[1], patch_size[2]
        d, h, w = img_size[0], img_size[1], img_size[2]
        assert (
            d % p_d == 0 and h % p_h == 0 and w % p_w == 0
        ), "Image dimensions must be divisible by patch size."

        # Reshape to (N, c, d//p_d, p_d, h//p_h, p_h, w//p_w, p_w)
        x = imgs.view(N, in_channels, d // p_d, p_d, h // p_h, p_h, w // p_w, p_w)

        # Rearrange to (N, d//p_d, h//p_h, w//p_w, p_d, p_h, p_w, c) then (N, L, p_d*p_h*p_w*c)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(N, -1, p_d * p_h * p_w * in_channels)
    else:
        raise ValueError("spatial_dims must be 2 or 3")
    return x


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: str,
    checkpoint_name: Optional[str] = None,
):
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    file_path = os.path.join(checkpoint_dir, "last_checkpoint.pytorch")
    torch.save(state, file_path)
    if checkpoint_name is not None:
        named_file_path = os.path.join(checkpoint_dir, checkpoint_name)
        _ = shutil.copyfile(file_path, named_file_path)


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    model_key: Optional[str] = "model_state_dict",
    optimizer_key: Optional[str] = "optimizer_state_dict",
):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location="cpu")
    if model_key is not None:
        _ = model.load_state_dict(state[model_key])
    else:
        _ = model.load_state_dict(state)

    if optimizer is not None:
        assert (
            optimizer_key is not None
        ), "optimizer_key must be provided if optimizer is given"
        optimizer.load_state_dict(state[optimizer_key])

    return state


def generate_test_images(
    img_size: Union[int, Tuple[int, ...]],
    num_images: int = 1,
    num_channels: int = 1,
    spatial_dims: int = 2,
    circle_centers: Optional[List[Tuple[int, ...]]] = None,
    circle_radius: Tuple[int, int] = (5, 15),
    axis_stretch: Optional[Tuple[float, ...]] = None,
    rotation_range: Optional[Tuple[float, float]] = None,
    noise_level: float = 0.3,
    circle_intensity: float = 0.8,
    background_mean: float = 0.2,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate images with background noise and circles (2D) or spheres (3D) of greater intensity.
    Supports ellipses/ellipsoids via axis stretching and random rotation augmentation.

    Parameters:
    -----------
    img_size : int or tuple of ints
        Size of the image. If int, same size for all spatial dimensions.
        If tuple, length must match spatial_dims.
    num_images : int
        Number of images to generate
    num_channels : int
        Number of channels in the output
    spatial_dims : int
        Number of spatial dimensions (2 for 2D circles, 3 for 3D spheres)
    circle_centers : list of tuples or None
        List of center coordinates for circles/spheres. Length must match num_images.
        If None, centers are randomly sampled for each image.
    circle_radius : tuple of (min, max)
        Range for randomly sampling circle/sphere radius for each image
    axis_stretch : tuple of floats or None
        Scale factors for each axis to create ellipses/ellipsoids.
        For 2D: (scale_y, scale_x), for 3D: (scale_z, scale_y, scale_x)
        If None, creates perfect circles/spheres (all scales = 1.0)
        Example: (1.0, 2.0) stretches x-axis by 2x in 2D
    rotation_range : tuple of (min_angle, max_angle) or None
        Range of rotation angles in degrees for random augmentation.
        For 2D: rotation in the image plane
        For 3D: rotation around the z-axis (in xy-plane)
        If None, no rotation is applied.
    noise_level : float
        Standard deviation of Gaussian noise
    circle_intensity : float
        Intensity value for the circle/sphere (0 to 1)
    background_mean : float
        Mean intensity of the background (0 to 1)
    seed : int or None
        Random seed for reproducibility

    Returns:
    --------
    images : np.ndarray
        Generated images of shape (N, C, H, W) for 2D or (N, C, D, H, W) for 3D
        For 3D, spheres span multiple slices across the depth dimension.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Parse img_size
    if isinstance(img_size, int):
        img_shape = tuple([img_size] * spatial_dims)
    else:
        img_shape = tuple(img_size)
        if len(img_shape) != spatial_dims:
            raise ValueError(
                f"img_size tuple length {len(img_shape)} must match spatial_dims {spatial_dims}"
            )

    # Set default axis stretch (no stretching)
    if axis_stretch is None:
        axis_stretch = tuple([1.0] * spatial_dims)
    else:
        if len(axis_stretch) != spatial_dims:
            raise ValueError(
                f"axis_stretch must have {spatial_dims} values, got {len(axis_stretch)}"
            )

    # Validate or generate circle centers
    if circle_centers is not None:
        if len(circle_centers) != num_images:
            raise ValueError(
                f"Number of circle_centers {len(circle_centers)} must match num_images {num_images}"
            )
        # Validate center dimensions
        for center in circle_centers:
            if len(center) != spatial_dims:
                raise ValueError(f"Each center must have {spatial_dims} coordinates")
    else:
        # Randomly sample centers
        circle_centers = []
        for _ in range(num_images):
            center = tuple(np.random.randint(0, dim_size) for dim_size in img_shape)
            circle_centers.append(center)

    # Initialize output array
    output_shape = (num_images, num_channels) + img_shape
    images = np.zeros(output_shape)

    # Generate each image
    for img_idx in range(num_images):
        # Sample random radius for this image
        radius = np.random.randint(circle_radius[0], circle_radius[1] + 1)

        # Sample random rotation angle if rotation_range is specified
        if rotation_range is not None:
            angle_deg = np.random.uniform(rotation_range[0], rotation_range[1])
            angle_rad = np.deg2rad(angle_deg)
        else:
            angle_rad = 0.0

        # Generate each channel
        for ch_idx in range(num_channels):
            # Create base image with background noise
            image = np.random.normal(background_mean, noise_level, img_shape)

            # Create circle/sphere mask with optional stretching and rotation
            center = circle_centers[img_idx]

            if spatial_dims == 2:
                # Create 2D ellipse with rotation
                y, x = np.ogrid[: img_shape[0], : img_shape[1]]

                # Center coordinates relative to circle center
                y_centered = y - center[0]
                x_centered = x - center[1]

                # Apply rotation
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                y_rot = cos_a * y_centered - sin_a * x_centered
                x_rot = sin_a * y_centered + cos_a * x_centered

                # Apply axis stretching and create ellipse mask
                mask = (
                    (y_rot / axis_stretch[0]) ** 2 + (x_rot / axis_stretch[1]) ** 2
                ) <= radius**2

            elif spatial_dims == 3:
                # Create 3D ellipsoid with rotation around z-axis
                z, y, x = np.ogrid[: img_shape[0], : img_shape[1], : img_shape[2]]

                # Center coordinates relative to sphere center
                z_centered = z - center[0]
                y_centered = y - center[1]
                x_centered = x - center[2]

                # Apply rotation around z-axis (rotates in xy-plane)
                cos_a = np.cos(angle_rad)
                sin_a = np.sin(angle_rad)
                y_rot = cos_a * y_centered - sin_a * x_centered
                x_rot = sin_a * y_centered + cos_a * x_centered
                z_rot = z_centered  # No rotation in z

                # Apply axis stretching and create ellipsoid mask
                mask = (
                    (z_rot / axis_stretch[0]) ** 2
                    + (y_rot / axis_stretch[1]) ** 2
                    + (x_rot / axis_stretch[2]) ** 2
                ) <= radius**2
            else:
                raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

            # Add circle/sphere with higher intensity
            num_pixels = int(np.sum(mask))
            if num_pixels > 0:
                image[mask] = circle_intensity + np.random.normal(
                    0, noise_level * 0.5, num_pixels
                )

            # Clip values to [0, 1] range
            image = np.clip(image, 0, 1)

            images[img_idx, ch_idx] = image

    return images

import glob
import importlib
from itertools import chain
import os
from numpy.typing import NDArray
import numpy as np
from typing import Any, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel

from MAE.network.logging import get_logger

logger = get_logger("Dataset")


def get_roi_slice(roi: Sequence[Tuple[int, ...]]):
    # Create a tuple of slice objects based on the input list
    slices = tuple(slice(start, stop) for start, stop in roi)
    return slices


def mirror_pad(image: NDArray[Any], padding_shape: Tuple[int, ...]) -> NDArray[Any]:
    """
    Pad the image with a mirror reflection of itself.

    This function is used on data in its original shape before it is split into patches.

    Args:
        image (np.ndarray): The input image array to be padded.
        padding_shape (tuple of int): Specifies the amount of padding for each dimension, should be YX or ZYX.

    Returns:
        np.ndarray: The mirror-padded image.

    Raises:
        ValueError: If any element of padding_shape is negative.
    """
    assert (
        len(padding_shape) == 3
    ), "Padding shape must be specified for each dimension: ZYX"

    if any(p < 0 for p in padding_shape):
        raise ValueError("padding_shape must be non-negative")

    if all(p == 0 for p in padding_shape):
        return image

    pad_width = [(p, p) for p in padding_shape]

    if image.ndim == 4:
        pad_width = [(0, 0)] + pad_width
    return np.pad(image, pad_width, mode="reflect")


class StatsParams(BaseModel):
    pmin: Optional[float]
    pmax: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    min_value: Optional[float]
    max_value: Optional[float]
    percentile_min: Optional[float]
    percentile_max: Optional[float]


def calculate_stats(
    img: Optional[Union[NDArray[Any], Sequence[NDArray[Any]]]],
    skip: bool = False,
    percentile_min: Optional[float] = None,
    percentile_max: Optional[float] = None,
) -> StatsParams:
    """
    Calculates the minimum percentile, maximum percentile, mean, and standard deviation of the image.

    Args:
        img: The input image array.
        skip: if True, skip the calculation and return None for all values.

    Returns:
        tuple[float, float, float, float]: The minimum percentile, maximum percentile, mean, and std dev
    """
    # if img is list, flatten and combine items of list
    if isinstance(img, list):
        img = np.concatenate([np.ravel(arr) for arr in img])
    if not skip:
        assert img is not None, "Image data must be provided if skip is False"
        mean = float(np.mean(img))
        std = float(np.std(img))
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        if percentile_min is not None:
            pmin = float(np.percentile(img, percentile_min))
        else:
            pmin = None
        if percentile_max is not None:
            pmax = float(np.percentile(img, percentile_max))
        else:
            pmax = None

    else:
        pmin, pmax, mean, std, min_val, max_val = None, None, None, None, None, None

    return StatsParams(
        pmin=pmin,
        pmax=pmax,
        mean=mean,
        std=std,
        min_value=min_val,
        max_value=max_val,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
    )


def get_class(class_name: str, modules: Sequence[str]):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f"Unsupported dataset class: {class_name}")


def loader_classes(class_name: str):
    modules = [
        "MAE.datasets.utils",
        "MAE.datasets.dataset",
        "MAE.datasets.slice_builders",
    ]
    return get_class(class_name, modules)


def traverse_h5_paths(file_paths: Sequence[str]) -> List[str]:
    results: List[str] = []
    for file_path in file_paths:
        if os.path.isdir(file_path):
            # if file path is a directory take all H5 files in that directory
            iters = [
                glob.glob(os.path.join(file_path, ext))
                for ext in ["*.h5", "*.hdf", "*.hdf5", "*.hd5"]
            ]
            for fp in chain(*iters):
                results.append(fp)
        else:
            results.append(file_path)
    return results


def return_dtype(str_dtype: str) -> np.dtype:
    """
    Converts string representation of a numpy dtype to actual numpy dtype.

    Args:
        str_dtype: string representation of a numpy dtype

    Returns:
        np.dtype: corresponding numpy dtype
    """
    try:
        dtype = np.dtype(str_dtype)
    except TypeError as e:
        raise ValueError(f"Unsupported dtype string: {str_dtype}") from e
    return dtype

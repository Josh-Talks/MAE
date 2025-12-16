import h5py  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from pydantic import BaseModel
from typing import Any, List, Literal, Optional, Tuple, Union

from MAE.utils import ensure_tuple, is_ndarray
from MAE.logging import get_logger
from .utils import mirror_pad, loader_classes

logger = get_logger("Dataset")


class SliceBuilderConfig(BaseModel):
    name: Literal["SliceBuilder", "SingleZSliceBuilder"]
    patch_shape: Tuple[int, ...]
    stride_shape: Tuple[int, ...]


class BaseShapeWrapper:
    """Base class for shape wrappers that provides common shape calculation logic."""

    def __init__(
        self,
        file_path: str,
        internal_path: str,
        roi: Optional[Tuple[slice[Any, Any, Any], ...]] = None,
        auto_padding: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        self.file_path = file_path
        self.internal_path = internal_path
        self.roi = roi
        self.auto_padding = auto_padding or (0, 0, 0)
        self._shape = None
        self._ndim = None
        self._needs_padding = any(p > 0 for p in self.auto_padding)

        # Pre-calculate shape at initialization for better performance
        self._calculate_shape()

    def _calculate_shape(self):
        """Calculate shape once at initialization."""
        with h5py.File(self.file_path, "r") as f:
            dataset = f[self.internal_path]
            assert isinstance(
                dataset, h5py.Dataset
            ), f"Expected Dataset, got {type(dataset)}"
            dataset_shape: Tuple[int, ...] = tuple(dataset.shape)  # pyright: ignore
            base_shape = dataset_shape
            if self.roi is not None:
                if isinstance(self.roi, tuple) and all(
                    isinstance(r, slice) for r in self.roi
                ):
                    base_shape = tuple(
                        (
                            len(range(*slice_obj.indices(int(dim_size))))
                            if slice_obj != slice(None)
                            else int(dim_size)
                        )
                        for slice_obj, dim_size in zip(self.roi, dataset_shape)
                    )
                elif isinstance(self.roi, (list, tuple)):
                    base_shape = (len(self.roi),) + dataset_shape[1:]

            # Apply padding to shape
            if self._needs_padding:
                if len(base_shape) == 4:  # 4D case (C, Z, Y, X)
                    self._shape = (
                        base_shape[0],
                        base_shape[1] + 2 * self.auto_padding[0],
                        base_shape[2] + 2 * self.auto_padding[1],
                        base_shape[3] + 2 * self.auto_padding[2],
                    )
                else:  # 3D case (Z, Y, X)
                    self._shape = (
                        base_shape[0] + 2 * self.auto_padding[0],
                        base_shape[1] + 2 * self.auto_padding[1],
                        base_shape[2] + 2 * self.auto_padding[2],
                    )
            else:
                self._shape = base_shape

            self._ndim = len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim


class DataAccessShapeWrapper(BaseShapeWrapper):
    """Shape wrapper with data access capability. Used by FilterSliceBuilder."""

    def __init__(
        self,
        file_path: str,
        internal_path: str,
        roi: Optional[Tuple[slice[Any, Any, Any], ...]] = None,
        auto_padding: Optional[Tuple[int, int, int]] = None,
    ):
        self._cached_data = None
        super().__init__(file_path, internal_path, roi, auto_padding)

    def _load_and_cache_data(self):
        """Load data once and cache it with padding applied."""
        if self._cached_data is None:
            with h5py.File(self.file_path, "r") as f:
                ds = f[self.internal_path]
                assert isinstance(ds, h5py.Dataset), f"Expected Dataset, got {type(ds)}"
                if self.roi is not None:
                    data = ds[self.roi]  # pyright: ignore[reportUnknownVariableType]
                else:
                    data = ds[:]  # pyright: ignore[reportUnknownVariableType]
                assert is_ndarray(data), "Data should be a numpy ndarray"
                # Apply automatic padding once and cache
                if self._needs_padding:
                    self._cached_data = mirror_pad(data, self.auto_padding)
                else:
                    self._cached_data = data
        return self._cached_data

    def __getitem__(self, idx: int) -> NDArray[Any]:
        """Load actual data when accessed - used by FilterSliceBuilder."""
        data = self._load_and_cache_data()
        return data[idx]


class ShapeOnlyWrapper(BaseShapeWrapper):
    """Lightweight wrapper that only provides shape info. Used by standard SliceBuilder."""

    def __getitem__(self, idx: int):
        """Data access not supported - StandardHDF5Dataset loads data directly."""
        raise RuntimeError(
            "ShapeOnlyWrapper is for shape info only. Data should be loaded by StandardHDF5Dataset."
        )


dataset_type = Union[ShapeOnlyWrapper, DataAccessShapeWrapper]


class SliceBuilder:
    """
    Builds the position of the patches in a given raw/label/weight ndarray based on the patch and stride shape.

    Args:
        raw_dataset (ndarray): raw data
        patch_shape (tuple): the shape of the patch DxHxW
        stride_shape (tuple): the shape of the stride DxHxW
        kwargs: additional metadata
    """

    def __init__(
        self,
        raw_dataset: BaseShapeWrapper,
        patch_shape: Tuple[int, ...],
        stride_shape: Tuple[int, ...],
    ):
        super().__init__()
        patch_shape = ensure_tuple(patch_shape, 3)
        stride_shape = ensure_tuple(stride_shape, 3)
        assert len(patch_shape) == 3, "patch_shape must be a 3D tuple"

        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)

    @property
    def raw_slices(self):
        return self._raw_slices

    @staticmethod
    def _build_slices(
        dataset: BaseShapeWrapper,
        patch_shape: Tuple[int, ...],
        stride_shape: Tuple[int, ...],
    ):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices: List[Tuple[slice, ...]] = []
        assert dataset.shape is not None, "Dataset shape must be known"
        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape  # pyright: ignore
        else:
            i_z, i_y, i_x = dataset.shape  # pyright: ignore

        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x),
                    )
                    if dataset.ndim == 4:
                        slice_idx = (
                            slice(0, in_channels),  # pyright: ignore
                        ) + slice_idx
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i: int, k: int, s: int):
        assert i >= k, "Sample size has to be bigger than the patch size"
        j = 0
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


class SingleZSliceBuilder:
    """
    Custom SliceBuilder that ensures exactly one patch per Z slice,
    centered in Y and X dimensions. Only returns patches where the patch_shape
    fits within the dataset dimensions.
    """

    def __init__(
        self,
        raw_dataset: BaseShapeWrapper,
        patch_shape: Tuple[int, ...],
        stride_shape: Tuple[int, ...],
    ):
        super().__init__()
        patch_shape = ensure_tuple(patch_shape, 3)
        stride_shape = ensure_tuple(stride_shape, 3)
        assert len(patch_shape) == 3, "patch_shape must be a 3D tuple"

        # Validate that patch fits in dataset dimensions
        self._validate_patch_fits(raw_dataset, patch_shape)

        # Build custom slices
        self._raw_slices = self._build_single_z_slices(
            raw_dataset, patch_shape, stride_shape
        )

    def _validate_patch_fits(
        self, dataset: BaseShapeWrapper, patch_shape: Tuple[int, ...]
    ):
        """Validate that the patch_shape fits within the dataset dimensions"""
        assert dataset.shape is not None, "Dataset shape must be known"
        if dataset.ndim == 4:
            _, i_z, i_y, i_x = dataset.shape  # pyright: ignore
        else:
            i_z, i_y, i_x = dataset.shape  # pyright: ignore

        k_z, k_y, k_x = patch_shape

        if i_z < k_z:
            raise ValueError(
                f"Patch size Z dimension ({k_z}) is larger than dataset Z dimension ({i_z})"
            )
        if i_y < k_y:
            raise ValueError(
                f"Patch size Y dimension ({k_y}) is larger than dataset Y dimension ({i_y})"
            )
        if i_x < k_x:
            raise ValueError(
                f"Patch size X dimension ({k_x}) is larger than dataset X dimension ({i_x})"
            )

    def _build_single_z_slices(
        self,
        dataset: BaseShapeWrapper,
        patch_shape: Tuple[int, ...],
        stride_shape: Tuple[int, ...],
    ):
        """Build slices ensuring exactly one centered patch per Z slice"""
        slices: List[Tuple[slice, ...]] = []

        if dataset.ndim == 4:
            in_channels, i_z, i_y, i_x = dataset.shape  # pyright: ignore
        else:
            i_z, i_y, i_x = dataset.shape  # pyright: ignore

        k_z, k_y, k_x = patch_shape
        s_z, _, _ = stride_shape

        assert (
            isinstance(i_z, int) and isinstance(i_y, int) and isinstance(i_x, int)
        ), "Dataset dimensions must be integers"

        # For Z dimension: use the original logic to get all Z positions
        z_steps = list(self._gen_indices(i_z, k_z, s_z))

        # For Y and X dimensions: Use centered positions only
        # Calculate center positions for Y and X dimensions
        y_pos = (i_y - k_y) // 2
        x_pos = (i_x - k_x) // 2

        # Ensure positions are non-negative (should be guaranteed by validation)
        y_pos = max(0, y_pos)
        x_pos = max(0, x_pos)

        # Build slices: exactly one centered Y position and one centered X position per Z
        for z in z_steps:
            slice_idx = (
                slice(z, z + k_z),
                slice(y_pos, y_pos + k_y),
                slice(x_pos, x_pos + k_x),
            )
            if dataset.ndim == 4:
                assert isinstance(
                    in_channels, int  # pyright: ignore
                ), "In channels must be an integer"
                slice_idx = (slice(0, in_channels),) + slice_idx
            slices.append(slice_idx)

        return slices

    @staticmethod
    def _gen_indices(i: int, k: int, s: int):
        assert i >= k, "Sample size has to be bigger than the patch size"
        j = 0
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


def get_slice_builder(raws: BaseShapeWrapper, config: SliceBuilderConfig):
    logger.info(f"Slice builder config: {config.model_dump()}")
    slice_builder_cls = loader_classes(config.name)
    return slice_builder_cls(raws, **config.model_dump(exclude={"name"}))

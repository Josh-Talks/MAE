from abc import abstractmethod
import h5py  # pyright: ignore[reportMissingTypeStubs]
from numpy.typing import NDArray
from pydantic import BaseModel
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from MAE.logging import get_logger
from MAE.utils import is_ndarray
from .utils import (
    calculate_stats,
    get_roi_slice,
    mirror_pad,
    traverse_h5_paths,
    loader_classes,
)
from .slice_builders import (
    ShapeOnlyWrapper,
    SliceBuilderConfig,
    get_slice_builder,
)
from .transformation import transforms_type, ImageTransformations

logger = get_logger("Dataset")


class DatasetConfig(BaseModel, frozen=True):
    name: Literal["StandardHDF5Dataset",]
    file_paths: Sequence[str]
    raw_internal_path: str
    global_normalization: bool
    global_percentiles: Optional[Tuple[float, float]]
    slice_builder: SliceBuilderConfig
    transformations: transforms_type
    roi: Optional[Sequence[Tuple[int, ...]]]


class LoaderConfig(BaseModel, frozen=True):
    batch_size: int
    num_workers: int
    dataset: DatasetConfig


default_loader: LoaderConfig = LoaderConfig(
    batch_size=32,
    num_workers=8,
    dataset=DatasetConfig(
        name="StandardHDF5Dataset",
        file_paths=[
            "/g/kreshuk/talks/data/Hmito/train_converted.h5",
        ],
        raw_internal_path="raw",
        global_normalization=True,
        global_percentiles=None,
        roi=((0, 150), (0, 1280), (0, 1280)),
        transformations={
            "raw": [
                {"name": "Normalize"},
                {"name": "ToTensor", "expand_dims": True},
            ],
        },
        slice_builder=SliceBuilderConfig(
            name="SliceBuilder",
            patch_shape=(1, 256, 256),
            stride_shape=(1, 256, 256),
        ),
    ),
)


class AbstractHDF5Dataset(Dataset[Any]):
    """
    Implementation of torch.utils.data.Dataset backed by the HDF5 files, which iterates over the raw and label datasets
    patch by patch with a given stride.

    Args:
        file_path (str): path to H5 file containing raw data as well as labels and per pixel weights (optional)
        phase (str): 'train' for training, 'val' for validation, 'test' for testing
        slice_builder_config (dict): configuration of the SliceBuilder
        transformer_config (dict): data augmentation configuration
        raw_internal_path (str or list): H5 internal path to the raw dataset
        label_internal_path (str or list): H5 internal path to the label dataset
        weight_internal_path (str or list): H5 internal path to the per pixel weights (optional)
        global_normalization (bool): if True, the mean and std of the raw data will be calculated over the whole dataset
    """

    def __init__(
        self,
        file_path: str,
        roi: Optional[Sequence[Tuple[int, ...]]],
        slice_builder_config: SliceBuilderConfig,
        transforms_config: transforms_type,
        raw_internal_path: str = "raw",
        global_normalization: bool = True,
        global_percentiles: Optional[Tuple[float, float]] = None,
        auto_padding: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()

        self.file_path = file_path
        if roi is not None:
            self.roi = get_roi_slice(roi)
        else:
            self.roi = roi
        self.raw_internal_path = raw_internal_path
        self.patch_shape = slice_builder_config.patch_shape
        self.auto_padding = auto_padding or (0, 0, 0)

        if global_normalization:
            logger.info("Calculating mean and std of the raw data...")
            with h5py.File(file_path, "r") as f:
                ds = f[raw_internal_path]
                assert isinstance(ds, h5py.Dataset)
                if self.roi is not None:
                    raw = ds[self.roi]  # pyright: ignore
                else:
                    raw = ds[:]  # pyright: ignore

                assert is_ndarray(raw), "Raw data should be a numpy ndarray"

                # Apply automatic padding for global normalization only if needed
                if any(p > 0 for p in self.auto_padding):
                    raw = mirror_pad(raw, self.auto_padding)

                if global_percentiles is not None:
                    stats = calculate_stats(
                        raw,
                        percentile_min=global_percentiles[0],
                        percentile_max=global_percentiles[1],
                    )
                else:
                    stats = calculate_stats(raw)
                print(f"Global mean: {stats.mean}, global std: {stats.std}")
        else:
            stats = calculate_stats(None, True)

        self.transformer = ImageTransformations(transforms_config, stats)
        self.raw_transform = self.transformer.raw_transform()

        raw_wrapper = ShapeOnlyWrapper(
            file_path, raw_internal_path, self.roi, self.auto_padding
        )

        assert raw_wrapper.ndim in [
            3,
            4,
        ], "Raw dataset must be 3D (DxHxW) or 4D (CxDxHxW)"

        # Build slice indices - SliceBuilder only uses .shape and .ndim
        slice_builder = get_slice_builder(raw_wrapper, slice_builder_config)
        self.raw_slices = slice_builder.raw_slices

        self.patch_count = len(self.raw_slices)
        logger.info(f"Number of patches: {self.patch_count}")

    @abstractmethod
    def get_raw_patch(self, idx: int) -> NDArray[Any]:
        raise NotImplementedError

    def volume_shape(self) -> Tuple[int, ...]:
        with h5py.File(self.file_path, "r") as f:
            raw_ds = f[self.raw_internal_path]
            assert isinstance(raw_ds, h5py.Dataset), "Raw dataset not found"
            if raw_ds.ndim == 3:
                return raw_ds.shape  # pyright: ignore
            else:
                return raw_ds.shape[1:]  # pyright: ignore

    def __getitem__(self, idx: int):
        if idx >= len(self):
            raise StopIteration

        raw_idx = self.raw_slices[idx]

        raw_patch_transformed = self.raw_transform(self.get_raw_patch(raw_idx))

        return raw_patch_transformed

    def __len__(self):
        return self.patch_count

    def get_patch_shape(self):
        return self.patch_shape

    @classmethod
    def create_datasets(cls, ds_cfg: DatasetConfig):
        # file_paths may contain both files and directories; if the file_path is a directory all H5 files inside
        # are going to be included in the final file_paths
        file_paths = traverse_h5_paths(ds_cfg.file_paths)

        datasets: List[AbstractHDF5Dataset] = []
        for file_path in file_paths:
            try:
                dataset = cls(
                    file_path=file_path,
                    roi=ds_cfg.roi,
                    slice_builder_config=ds_cfg.slice_builder,
                    transforms_config=ds_cfg.transformations,
                    raw_internal_path=ds_cfg.raw_internal_path,
                    global_normalization=ds_cfg.global_normalization,
                    global_percentiles=ds_cfg.global_percentiles,
                )
                datasets.append(dataset)
            except Exception:
                logger.error(f"Skipping: {file_path}", exc_info=True)
        return datasets


class StandardHDF5Dataset(AbstractHDF5Dataset):
    """
    Implementation of the HDF5 dataset which loads the data from the H5 files into the memory.
    Fast but might consume a lot of memory.
    """

    def __init__(
        self,
        file_path: str,
        roi: Optional[Sequence[Tuple[int, ...]]],
        slice_builder_config: SliceBuilderConfig,
        transforms_config: transforms_type,
        raw_internal_path: str = "raw",
        global_normalization: bool = True,
        global_percentiles: Optional[Tuple[float, float]] = None,
    ):

        # Calculate automatic padding before calling parent constructor
        if roi is not None:
            roi_slice = get_roi_slice(roi)
        else:
            roi_slice = roi
        self.auto_padding = tuple(
            self._calculate_auto_padding(
                file_path, raw_internal_path, roi_slice, slice_builder_config
            )
        )
        self._needs_padding = any(p > 0 for p in self.auto_padding)

        super().__init__(
            file_path=file_path,
            roi=roi,
            slice_builder_config=slice_builder_config,
            transforms_config=transforms_config,
            raw_internal_path=raw_internal_path,
            global_normalization=global_normalization,
            global_percentiles=global_percentiles,
            auto_padding=self.auto_padding,
        )
        self._raw = None
        self._raw_padded = None
        self._label = None
        self._weight_map = None

    def _calculate_auto_padding(
        self,
        file_path: str,
        raw_internal_path: str,
        roi: Optional[Tuple[slice[Any, Any, Any], ...]],
        slice_builder_config: SliceBuilderConfig,
    ):
        """Calculate automatic padding needed for patch extraction."""
        patch_shape = slice_builder_config.patch_shape

        # Get actual volume shape
        with h5py.File(file_path, "r") as f:
            dataset = f[raw_internal_path]
            assert isinstance(dataset, h5py.Dataset), "Raw dataset not found"
            dataset_shape: Tuple[int, ...] = tuple(dataset.shape)  # pyright: ignore
            if roi is not None:
                if isinstance(roi, tuple) and all(isinstance(r, slice) for r in roi):
                    # Calculate shape for slice-based ROI
                    volume_shape = tuple(
                        (
                            len(range(*slice_obj.indices(int(dim_size))))
                            if slice_obj != slice(None)
                            else int(dim_size)
                        )
                        for slice_obj, dim_size in zip(roi, dataset_shape)
                    )
                elif isinstance(roi, (list, tuple)):
                    # Handle index-based ROI (first dimension)
                    volume_shape = (len(roi),) + dataset_shape[1:]
            else:
                volume_shape = dataset_shape
            # Handle 4D case (remove channel dimension)
            if len(volume_shape) == 4:
                volume_shape = volume_shape[1:]  # Remove channel dimension

        # Calculate required padding for each spatial dimension
        padding: List[int] = []
        for patch_dim, vol_dim in zip(patch_shape, volume_shape):
            if patch_dim > vol_dim:
                # Calculate padding needed to make volume at least as large as patch
                needed_size = patch_dim
                current_size = vol_dim
                total_padding = needed_size - current_size
                # Split padding evenly on both sides (mirror_pad expects per-side padding)
                padding_per_side = (
                    total_padding + 1
                ) // 2  # Round up to ensure sufficient padding
                padding.append(padding_per_side)
            else:
                padding.append(0)

        logger.info(
            f"Auto-calculated padding: {padding} for patch_shape: {patch_shape} and volume_shape: {volume_shape}"
        )
        return padding

    def get_raw_patch(self, idx: int):
        if self._raw is None:
            with h5py.File(self.file_path, "r") as f:
                assert (
                    self.raw_internal_path in f
                ), f"Dataset {self.raw_internal_path} not found in {self.file_path}"
                ds = f[self.raw_internal_path]
                assert isinstance(
                    ds, h5py.Dataset
                ), f"Dataset {self.raw_internal_path} is not a valid H5 dataset"
                if self.roi is not None:
                    raw_data = ds[self.roi]  # pyright: ignore
                else:
                    raw_data = ds[:]  # pyright: ignore

                assert is_ndarray(raw_data), "Raw data should be a numpy ndarray"

                # Apply automatic padding once and cache
                if self._needs_padding:
                    self._raw = mirror_pad(raw_data, self.auto_padding)
                else:
                    self._raw = raw_data
        assert is_ndarray(self._raw), "Raw data should be a numpy ndarray"
        return self._raw[idx]


def get_pretrain_loader(config: LoaderConfig):
    dataset_class = loader_classes(config.dataset.name)

    train_datasets: List[Dataset[Any]] = dataset_class.create_datasets(config.dataset)

    num_workers = config.num_workers
    logger.info(f"Number of workers for train/val dataloader: {num_workers}")
    batch_size = config.batch_size

    if torch.cuda.device_count() > 1:
        logger.info(
            f"{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}"
        )
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f"Batch size for train loader: {batch_size}")

    train_loader: DataLoader[Any] = DataLoader(
        ConcatDataset(train_datasets),
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        shuffle=True,
    )

    return train_loader

import numpy as np
from numpy.typing import NDArray
import importlib
import torch
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

GLOBAL_RANDOM_STATE = np.random.RandomState(47)

from .utils import StatsParams, return_dtype

aug_config_type = Mapping[
    str, Optional[Union[str, bool, int, Sequence[int], Sequence[float]]]
]

transforms_type = Mapping[
    str,
    List[aug_config_type],
]


class Compose(object):
    def __init__(self, transforms: List[Callable[[NDArray[Any]], NDArray[Any]]]):
        super().__init__()
        self.transforms = transforms

    def __call__(self, m: NDArray[Any]) -> NDArray[Any]:
        for t in self.transforms:
            m = t(m)
        return m


class PercentileNormalizer:
    def __init__(
        self,
        pmin: Optional[float] = None,
        pmax: Optional[float] = None,
        percentile_min: float = 1,
        percentile_max: float = 99.6,
        channelwise: bool = False,
        eps: float = 1e-10,
        **kwargs: Any,
    ):
        super().__init__()
        self.eps = eps
        self.pmin = pmin
        self.pmax = pmax
        self.percentile_min = percentile_min
        self.percentile_max = percentile_max
        self.channelwise = channelwise

    def __call__(self, m: NDArray[Any]) -> NDArray[Any]:
        if (self.pmin) is not None:
            assert self.pmax is not None, "pmax must be provided"
            pmin, pmax = self.pmin, self.pmax
        else:
            if self.channelwise:
                axes = list(range(m.ndim))
                # average across channels
                axes = tuple(axes[1:])
                pmin = np.percentile(m, self.percentile_min, axis=axes, keepdims=True)
                pmax = np.percentile(m, self.percentile_max, axis=axes, keepdims=True)
            else:
                pmin = np.percentile(m, self.percentile_min)
                pmax = np.percentile(m, self.percentile_max)

        return (m - pmin) / (pmax - pmin + self.eps)


class Normalize:
    """
    Apply simple min-max scaling to a given input tensor, i.e. shrinks the range of the data
    in a fixed range of [-1, 1] or in case of norm01==True to [0, 1]. In addition, data can be
    clipped by specifying min_value/max_value either globally using single values or via a
    list/tuple channelwise if enabled.
    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        norm01: bool = True,
        channelwise: bool = False,
        eps: float = 1e-10,
        **kwargs: Any,
    ):
        super().__init__()
        if min_value is not None and max_value is not None:
            assert max_value > min_value
        self.min_value = min_value
        self.max_value = max_value
        self.norm01 = norm01
        self.channelwise = channelwise
        self.eps = eps

    def __call__(self, m: NDArray[Any]) -> NDArray[Any]:

        if self.min_value is None:
            min_value = np.min(m)
        else:
            min_value = self.min_value

        if self.max_value is None:
            max_value = np.max(m)
        else:
            max_value = self.max_value

        # calculate norm_0_1 with min_value / max_value with the same dimension
        # in case of channelwise application
        norm_0_1 = (m - min_value) / (max_value - min_value + self.eps)

        if self.norm01 is True:
            return np.clip(norm_0_1, 0, 1)
        else:
            return np.clip(2 * norm_0_1 - 1, -1, 1)


class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state: np.random.RandomState, **kwargs: Any):
        super().__init__()
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m: NDArray[Any]) -> NDArray[Any]:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m


class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(
        self, random_state: np.random.RandomState, axis_prob: float = 0.5, **kwargs: Any
    ):
        super().__init__()
        assert random_state is not None, "RandomState cannot be None"
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m: NDArray[Any]) -> NDArray[Any]:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m


class ToTensor:
    """
    Converts a given input numpy.ndarray into torch.Tensor.

    Args:
        expand_dims (bool): if True, adds a channel dimension to the input data
        dtype (np.dtype): the desired output data type
    """

    def __init__(self, expand_dims: bool, dtype: str = "float32", **kwargs: Any):
        super().__init__()
        self.expand_dims = expand_dims
        self.dtype = return_dtype(dtype)

    def __call__(self, m: NDArray[Any]) -> torch.Tensor:
        assert m.ndim in [3, 4], "Supports only 3D (DxHxW) or 4D (CxDxHxW) images"
        # add channel dimension
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)

        return torch.from_numpy(m.astype(dtype=self.dtype))


class ImageTransformations:
    def __init__(self, transforms_config: transforms_type, stats_config: StatsParams):
        super().__init__()
        self.transforms_config = transforms_config
        self.config_base = stats_config.model_dump()
        self.seed = GLOBAL_RANDOM_STATE.randint(10000000)

    def raw_transform(self):
        return self._create_transform("raw")

    def label_transform(self):
        return self._create_transform("label")

    def weight_transform(self):
        return self._create_transform("weight")

    @staticmethod
    def _transformer_class(class_name: str):
        m = importlib.import_module("MAE.datasets.transformation")
        clazz = getattr(m, class_name)
        return clazz

    def _create_transform(self, name: str) -> Compose:
        assert name in self.transforms_config, f"Could not find {name} transform"
        return Compose(
            [self._create_augmentation(c) for c in self.transforms_config[name]]
        )

    def _create_augmentation(self, c: aug_config_type):
        config = dict(self.config_base)
        config.update(c)
        config["random_state"] = np.random.RandomState(self.seed)
        aug_class = self._transformer_class(config["name"])
        return aug_class(**config)

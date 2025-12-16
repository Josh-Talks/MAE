import numpy as np
from numpy.typing import NDArray
import importlib
from typing import Any, Callable, List, Mapping, Optional, Sequence, Union

GLOBAL_RANDOM_STATE = np.random.RandomState(47)

from .utils import StatsParams

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


class ImageTransformations:
    def __init__(self, transforms_config: transforms_type, stats_config: StatsParams):
        super().__init__()
        self.transforms_config = transforms_config
        self.config_base = stats_config
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

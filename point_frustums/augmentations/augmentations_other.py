from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import torch

from point_frustums.utils.targets import Targets
from .augmentations import Augmentation


@dataclass(frozen=True)
class NormalizationParameters:
    channels: list[int]
    mean: list[float]
    std: list[float]

    @cached_property
    def tensor_mean(self) -> torch.Tensor:
        return torch.tensor(self.mean)

    @cached_property
    def tensor_std(self) -> torch.Tensor:
        return torch.tensor(self.std)


class Normalize(Augmentation):
    def __init__(
        self,
        lidar: Optional[NormalizationParameters] = None,
        camera: Optional[NormalizationParameters] = None,
        radar: Optional[NormalizationParameters] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._lidar = lidar
        self._camera = camera
        self._radar = radar
        for modality in self.apply_to:
            assert getattr(self, f"_{modality}") is not None

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        data[:, self._camera.channels] = (
            data[:, self._camera.channels].sub(self._camera.tensor_mean).div(self._camera.tensor_std)
        )
        return data

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        data[:, self._lidar.channels] = (
            data[:, self._lidar.channels].sub(self._lidar.tensor_mean).div(self._lidar.tensor_std)
        )
        return data

    def targets(self, targets: Targets):
        return targets

    def metadata(self, metadata: MutableMapping):
        return metadata

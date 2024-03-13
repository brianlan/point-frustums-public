from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property
from math import ceil
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

    @property
    def log(self):
        return {"mean": self.tensor_mean, "std": self.tensor_std, "channels": self.channels}


def de_normalize(data: torch.Tensor, channels: list[int], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data[:, channels] = data[:, channels].mul(std).add(mean)
    return data


def normalize(data: torch.Tensor, channels: list[int], mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    data[:, channels] = data[:, channels].sub(mean).div(std)
    return data


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
        metadata["augmentations"][str(self)] = {"camera": self._camera.log}
        return normalize(data, self._camera.channels, self._camera.tensor_mean, self._camera.tensor_std)

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        metadata["augmentations"][str(self)] = {"lidar": self._lidar.log}
        return normalize(data, self._lidar.channels, self._lidar.tensor_mean, self._lidar.tensor_std)

    def targets(self, targets: Targets):
        return targets

    def metadata(self, metadata: MutableMapping):
        return metadata


class SubsampleData(Augmentation):
    """
    Randomly subsample the data.
    """

    def __init__(self, drop_ratio, n_max=1_000_000, **kwargs):
        super().__init__(**kwargs)
        assert 0 <= drop_ratio <= 1
        self._keep_ratio = 1 - drop_ratio
        self._n_max = n_max
        self._rng = torch.Generator()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        """
        Remove a fraction of the points.
        :param data:
        :param metadata:
        :return:
        """
        n_points, n_dims = data.shape
        indices_full = torch.randperm(n_points, generator=self._rng)
        n_points = ceil(self._keep_ratio * (min(n_points, self._n_max)))
        indices = indices_full[:n_points]
        return data[indices, ...]

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def targets(self, targets: Targets):
        raise SyntaxError("Subsampling should not be applied to targets.")

    def metadata(self, metadata: MutableMapping):
        return metadata

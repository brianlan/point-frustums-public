from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import cached_property
from typing import Optional
from math import ceil

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

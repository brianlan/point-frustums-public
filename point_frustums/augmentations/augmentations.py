import random
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from typing import Optional

import torch
from torch import nn

from point_frustums.utils.targets import Targets


class Augmentation(nn.Module, ABC):
    def __init__(self, apply_to: set):
        super().__init__()
        self.apply_to = apply_to

    def __str__(self):
        return str(self.__class__.__name__)

    @abstractmethod
    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        pass

    @abstractmethod
    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        pass

    @abstractmethod
    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        pass

    @abstractmethod
    def targets(self, targets: Targets):
        pass

    @abstractmethod
    def metadata(self, metadata: MutableMapping):
        pass

    def forward(  # pylint: disable=too-many-arguments
        self,
        lidar: Optional[MutableMapping[str, torch.Tensor]],
        camera: Optional[MutableMapping[str, torch.Tensor]],
        radar: Optional[MutableMapping[str, torch.Tensor]],
        targets: Optional[Targets],
        metadata: Optional[MutableMapping],
    ):
        """
        Apply the augmentation to all the provided data and targets.
        :param lidar:
        :param camera:
        :param radar:
        :param targets:
        :param metadata:
        :return:
        """

        if metadata is None:
            metadata = {}

        # Log the applied augmentation in the metadata
        if "augmentations" not in metadata:
            metadata["augmentations"] = {}

        metadata["augmentations"][str(self)] = {}
        metadata = self.metadata(metadata)

        if lidar is not None and "lidar" in self.apply_to:
            for key, value in lidar.items():
                lidar[key] = self.lidar(value, metadata)

        if camera is not None and "camera" in self.apply_to:
            for key, value in camera.items():
                camera[key] = self.camera(value, metadata)

        if radar is not None and "radar" in self.apply_to:
            for key, value in radar.items():
                radar[key] = self.radar(value, metadata)

        if targets is not None and "targets" in self.apply_to:
            targets = self.targets(targets)

        return lidar, camera, radar, targets, metadata


class RandomAugmentation(Augmentation, ABC):
    def __init__(self, probability: float, **kwargs):
        super().__init__(**kwargs)
        self.probability = probability
        self.apply: bool = True

    @abstractmethod
    def _refresh(self):
        pass

    def refresh(self):
        self.apply = random.random() < self.probability
        self._refresh()

    def forward(  # pylint: disable=too-many-arguments
        self,
        lidar: Optional[MutableMapping[str, torch.Tensor]],
        camera: Optional[MutableMapping[str, torch.Tensor]],
        radar: Optional[MutableMapping[str, torch.Tensor]],
        targets: Optional[Targets],
        metadata: Optional[MutableMapping],
    ):
        if self.apply:
            lidar, camera, radar, targets, metadata = super().forward(
                lidar=lidar, camera=camera, radar=radar, targets=targets, metadata=metadata
            )
        return lidar, camera, radar, targets, metadata

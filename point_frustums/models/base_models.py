from typing import Optional
from collections.abc import Mapping, MutableMapping
from abc import ABCMeta, abstractmethod

from torch import nn, Tensor


class Backbone(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_lidar(self, lidar: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_camera(self, camera: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_radar(self, camera: Tensor) -> Tensor:
        pass

    def forward(
        self,
        lidar: Optional[MutableMapping[str, Tensor]] = None,
        camera: Optional[MutableMapping[str, Tensor]] = None,
        radar: Optional[MutableMapping[str, Tensor]] = None,
    ) -> tuple[
        Optional[MutableMapping[str, Tensor]],
        Optional[MutableMapping[str, Tensor]],
        Optional[MutableMapping[str, Tensor]],
    ]:
        if lidar is not None:
            lidar = {sensor: self.forward_lidar(data) for sensor, data in lidar.items()}

        if camera is not None:
            camera = {sensor: self.forward_camera(data) for sensor, data in camera.items()}

        if radar is not None:
            radar = {sensor: self.forward_radar(data) for sensor, data in radar.items()}

        return lidar, camera, radar


class Neck(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        lidar: Optional[MutableMapping[str, Tensor]],
        camera: Optional[MutableMapping[str, Tensor]],
        radar: Optional[MutableMapping[str, Tensor]],
    ) -> Tensor | Mapping[str, Tensor]:
        pass


class Head(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, features: Tensor | Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        pass


class Detection3DModel(nn.Module):
    backbone: Backbone
    neck: Neck
    head: Head

    def forward(
        self,
        lidar: Optional[MutableMapping[str, Tensor]] = None,
        camera: Optional[MutableMapping[str, Tensor]] = None,
        radar: Optional[MutableMapping[str, Tensor]] = None,
    ):
        lidar, camera, radar = self.backbone(lidar=lidar, camera=camera, radar=radar)
        features = self.neck(lidar=lidar, camera=camera, radar=radar)
        return self.head(features=features)

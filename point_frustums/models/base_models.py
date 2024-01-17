from typing import Optional
from abc import ABCMeta, abstractmethod

from torch import nn, Tensor


class Backbone(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward_lidar(self, lidar: list[Tensor]) -> Tensor:
        pass

    @abstractmethod
    def forward_camera(self, camera: Tensor) -> Tensor:
        pass

    @abstractmethod
    def forward_radar(self, camera: Tensor) -> Tensor:
        pass

    def forward(
        self,
        lidar: dict[str, list[Tensor]],
        camera: Optional[dict[str, Tensor]] = None,
        radar: Optional[dict[str, Tensor]] = None,
    ) -> tuple[dict[str, Tensor], Optional[dict[str, Tensor]], Optional[dict[str, Tensor]],]:
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
        lidar: dict[str, Tensor],
        camera: Optional[dict[str, Tensor]],
        radar: Optional[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        pass


class Head(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, features: Tensor | dict[str, Tensor]) -> dict[str, Tensor]:
        pass


class Detection3DModel(nn.Module):
    def __init__(self, backbone: Backbone, neck: Neck, head: Head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(
        self,
        lidar: dict[str, list[Tensor]],
        camera: Optional[dict[str, Tensor]] = None,
        radar: Optional[dict[str, Tensor]] = None,
    ) -> dict[str, dict[str, Tensor]]:
        lidar, camera, radar = self.backbone(lidar=lidar, camera=camera, radar=radar)
        features = self.neck(lidar=lidar, camera=camera, radar=radar)
        return self.head(features=features)

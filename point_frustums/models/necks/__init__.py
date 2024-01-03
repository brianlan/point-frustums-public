from typing import Optional
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass

import torch
from torch import nn, Tensor


from ..base_models import Neck
from .spherical_coos_fpn import FPN


class MapCameraFeatures(nn.Module):  # pylint: disable=W0223
    pass


class MapRadarFeatures(nn.Module):  # pylint: disable=W0223
    pass


class PointFrustumsNeck(Neck):
    """The neck for the PointFrustums model.

    Support for camera/radar to lidar fusion is planned in the following way:
    1. Create models that map the camera or radar data to the lidar featuremap
        - Initialize the respective model for each active sensor
        - The initialized model is aware of the sensor extrinsic parameters required to map between featuremaps
        - Information is mapped to the lidar featuremap geometrically or by attention
        - Attention can be evaluated regionally to reduce computational the cost (xFormers masked attention?)
    2. For each modality, a featuremap is initialized with zeros
    3. The mapped features from active sensors are filled into the featuremap (average overlapping regions)
    4. The featuremaps from the modalities are concatenated and processed by channel-wise convolution
    """

    def __init__(
        self,
        target_sensor: str,
        fpn: FPN,
        map_camera: Optional[Mapping[MapCameraFeatures]] = None,
        map_radar: Optional[Mapping[MapRadarFeatures]] = None,
    ):
        super().__init__()
        self.target_sensor = target_sensor
        assert map_camera is None, "Camera to LiDAR fusion is not yet implemented."
        assert map_radar is None, "Radar to LiDAR fusion is not yet implemented."
        self.fpn = fpn
        self.map_camera = map_camera  # Feature mapping module for camera data
        self.map_radar = map_radar  # Feature mapping module for radar data

    def forward(
        self,
        lidar: MutableMapping[str, Tensor],
        camera: Optional[MutableMapping[str, Tensor]] = None,
        radar: Optional[MutableMapping[str, Tensor]] = None,
    ) -> Tensor | Mapping[str, Tensor]:
        featuremap = [lidar[self.target_sensor]]
        # TODO: Implement the mapping in case multiple LiDAR sensors are used
        # TODO: Implement the fusion functions; have to be executed for each sensor
        featuremap = torch.cat(featuremap, dim=-1)
        return self.fpn(featuremap)

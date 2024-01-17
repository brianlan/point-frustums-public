from torch import Tensor
from ..base_models import Backbone
from .frustum_encoder import FrustumEncoder


class PointFrustumsBackbone(Backbone):
    def __init__(self, lidar: FrustumEncoder, camera=None, radar=None):
        super().__init__()
        self.lidar = lidar
        self.camera = camera
        self.radar = radar

    def forward_lidar(self, lidar: list[Tensor]) -> Tensor:
        return self.lidar(lidar)

    def forward_camera(self, camera: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_radar(self, camera: Tensor) -> Tensor:
        raise NotImplementedError

from typing import Optional
from collections.abc import Mapping, MutableMapping

from torch import Tensor
from ..base_models import Neck


class PointFrustumsNeck(Neck):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lidar: Optional[MutableMapping[str, Tensor]],
        camera: Optional[MutableMapping[str, Tensor]],
        radar: Optional[MutableMapping[str, Tensor]],
    ) -> Tensor | Mapping[str, Tensor]:
        pass

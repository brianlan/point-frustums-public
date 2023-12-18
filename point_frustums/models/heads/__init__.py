from typing import NamedTuple
from collections.abc import Mapping

from torch import Tensor
from ..base_models import Head


class PointFrustumsHead(Head):
    def forward(self, features: Tensor | Mapping[str, Tensor]) -> NamedTuple:
        pass

from typing import Literal, TypeAlias

from torch import Tensor

Targets: TypeAlias = dict[Literal["class", "center", "wlh", "orientation", "attribute", "velocity"], Tensor]
Boxes: TypeAlias = dict[Literal["class", "center", "wlh", "orientation", "attribute", "velocity", "score"], Tensor]

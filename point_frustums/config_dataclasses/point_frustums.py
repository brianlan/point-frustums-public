from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from point_frustums.utils import geometry


@dataclass(slots=True, frozen=False)
class ModelOutputSpecification:
    strides: list[tuple[int, int]]
    layer_sizes: list[tuple[int, int]]
    layer_sizes_flat: list[int]
    total_size: int


@dataclass(slots=True, frozen=True)
class Predictions:
    score_vfl_factor: float = 0.3
    score_threshold: float = 0.1
    top_k: int = 1_000
    n_detections: int = 100
    nms_threshold: float = 0.6


@dataclass(frozen=True, slots=True)
class Loss:
    active: bool = True
    weight: float = 1.0
    kwargs: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Losses:
    label: Loss
    attribute: Loss
    vfl: Loss
    center_radial: Loss = Loss(kwargs={"beta": 0.2})
    center_angular: Loss = Loss(kwargs={"beta": 0.5})
    wlh: Loss = Loss()
    orientation: Loss = Loss(kwargs={"beta": 0.1})
    velocity: Loss = Loss()

    @property
    def losses(self) -> tuple[str, ...]:
        return self.__slots__  # pylint: disable=no-member


@dataclass(frozen=True)
class TargetAssignment:
    # Parametrize the weights of {classification, IoU, radial distance}
    alpha: float = 1.2
    beta: float = 1.0
    gamma: float = 1.0

    # Scaling factor applied to the center distance before tanh
    kappa: float = 4.0

    # Minimum number of predictions assigned to each target
    min_k: int = 2
    # Limit the supply (maximum number of predictions assigned to each target)
    max_k: int = 20

    # Stopping criterion for the Sinkhorn iteration
    threshold: float = 1e-5
    # Scaling factor used in the Sinkhorn iteration
    epsilon: float = 2e-1
    # The maximum number of iterations to perform
    max_iter: int = 2500


@dataclass(slots=True, kw_only=True)
class ConfigDiscretize:
    n_splits_azi: int
    n_splits_pol: int
    fov_azi_deg: tuple[int, int]
    fov_pol_deg: tuple[int, int]

    @property
    def fov_azi(self):
        return tuple(geometry.deg_to_rad(x) for x in self.fov_azi_deg)

    @property
    def fov_pol(self):
        return tuple(geometry.deg_to_rad(x) for x in self.fov_pol_deg)

    @property
    def range_pol(self):
        return self.fov_pol[1] - self.fov_pol[0]

    @property
    def range_azi(self):
        return self.fov_azi[1] - self.fov_azi[0]

    @property
    def delta_pol(self):
        return self.range_pol / self.n_splits_pol

    @property
    def delta_azi(self):
        return self.range_azi / self.n_splits_azi

    @property
    def n_splits(self):
        return self.n_splits_azi * self.n_splits_pol


@dataclass(slots=True, kw_only=True)
class DecoratorFunction:
    id: Literal["relative_angle", "distance_to_mean"]
    channel: Literal["x", "y", "z", "radial", "azimuthal", "polar", "intensity", "timestamp"]
    std: float


@dataclass(slots=True, kw_only=True)
class ConfigDecorate:
    functions: Sequence[DecoratorFunction]
    channels_out: Sequence[str]

    @property
    def n_channels_out(self):
        return len(self.channels_out)


@dataclass(slots=True, kw_only=True)
class ConfigVectorize:
    layers: Sequence[int]

    @property
    def n_channels_out(self):
        if len(self.layers) == 0:
            return None
        return self.layers[-1]


@dataclass(slots=True, kw_only=True)
class ConfigReduce:
    layers: Sequence[int]

    @property
    def n_channels_out(self):
        if len(self.layers) == 0:
            return None
        return self.layers[-1]


@dataclass(slots=True, kw_only=True)
class ConfigTransformerFrustumEncoder:
    n_channels_embedding: int
    n_channels_projection: int
    n_heads: int = 1
    n_encoders: int = 1
    dropout: float = 0.1

from dataclasses import dataclass
from enum import Enum
from collections.abc import Sequence

from functools import cached_property
from itertools import chain
from typing import Literal, Optional


class Labels:
    def __init__(self, labels: Sequence):
        self.labels = Enum("labels", labels, start=0)

    def from_index(self, index: int) -> Enum:
        return self.labels(index)

    def from_name(self, name: str) -> Enum:
        return self.labels[name]


@dataclass(frozen=True)
class Annotations:
    coos: str
    visibility: float
    class_aliases: list[str]
    alias_to_class: dict[str, str]
    alias_to_category: dict[str, str]
    category_to_attributes: dict[str, list]

    @cached_property
    def classes(self) -> Labels:
        return Labels(self.class_aliases)

    @cached_property
    def attributes(self) -> Labels:
        _attributes = list(set(chain.from_iterable(self.category_to_attributes.values())))
        return Labels(_attributes)

    @cached_property
    def class_to_alias(self) -> dict[str, str]:
        return {value: key for key, value in self.alias_to_class.items()}

    @cached_property
    def category_to_alias(self) -> dict[str, str]:
        return {value: key for key, value in self.alias_to_category.items()}

    def retrieve_class(self, class_name_verbose: str) -> Optional[Enum]:
        for class_alias, class_name_abbreviated in self.alias_to_class.items():
            if class_name_verbose.startswith(class_name_abbreviated):
                return self.classes.from_name(class_alias)
        return None


@dataclass(slots=True, frozen=True)
class Sensor:
    active: bool
    modality: Literal["lidar", "camera", "radar"]
    channels_out: list[str]
    angle_of_view_degrees: dict[Literal["polar", "azimuthal"], int]
    orientation_degrees: dict[Literal["polar", "azimuthal"], int]
    sweeps: Optional[int] = None
    resolution: Optional[dict[Literal["polar", "azimuthal"], int]] = None


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    name: str
    version: str
    sensors: dict[str, Sensor]
    annotations: Annotations
    load_velocity: bool = False
    load_can: bool = False

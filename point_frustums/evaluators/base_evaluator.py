from typing import Any
from abc import ABCMeta, abstractmethod

from point_frustums.config_dataclasses.dataset import DatasetConfig


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, dataset: DatasetConfig, meta: dict, evaluation_dir: str):
        pass

    @abstractmethod
    def parse(self, detections: Any, metadata: Any):
        pass

    @abstractmethod
    def serialize(self) -> dict:
        pass

    @abstractmethod
    def evaluate(self, epoch: int, split: str = "val"):
        pass

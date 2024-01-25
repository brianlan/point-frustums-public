from typing import Optional
from collections.abc import MutableMapping
from abc import ABCMeta, abstractmethod

from torch import Tensor
from pytorch_lightning import LightningModule

from point_frustums.models.base_models import Detection3DModel
from point_frustums.config_dataclasses.dataset import DatasetConfig


class Detection3DRuntime(LightningModule, metaclass=ABCMeta):
    def __init__(self, *args, model: Detection3DModel, dataset: DatasetConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset

    def forward(  # pylint: disable=arguments-differ
        self,
        lidar: dict[str, list[Tensor]],
        camera: Optional[MutableMapping[str, Tensor]],
        radar: Optional[MutableMapping[str, Tensor]],
    ):
        return self.model.forward(lidar=lidar, camera=camera, radar=radar)

    @abstractmethod
    def training_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def validation_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

    @abstractmethod
    def test_step(self, batch, batch_idx):  # pylint: disable=arguments-differ
        pass

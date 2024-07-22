from loguru import logger
from pytorch_lightning import Callback, Trainer
from torch import load

from point_frustums.models import Detection3DRuntime


class ModelWeightsFromCheckpoint(Callback):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def on_fit_start(self, trainer: Trainer, pl_module: Detection3DRuntime) -> None:
        logger.info(f"Initializing the model weights from the checkpoint '{self.checkpoint_path}'")
        pl_module.load_state_dict(load(self.checkpoint_path)["state_dict"])

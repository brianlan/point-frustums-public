import inspect
from typing import Sequence

import torch
from loguru import logger
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.utilities import move_data_to_device
from torch import nn

from point_frustums.models import Detection3DRuntime


class ModelWeightsFromCheckpoint(Callback):
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def on_fit_start(self, trainer: Trainer, pl_module: Detection3DRuntime) -> None:
        logger.info(f"Initializing the model weights from the checkpoint '{self.checkpoint_path}'")
        pl_module.load_state_dict(torch.load(self.checkpoint_path)["state_dict"])


class WeightInitLSUV(Callback):
    """
    A Callback that applies Layer-Sequential UnitVariance (LSUV) initialization to the model.
    Reference: arXiv:1511.06422v7
    """

    def __init__(
        self,
        init_types: Sequence[str],
        t_max=15,
        std_tol=0.01,
        verbose=False,
    ):
        """
        Initialize the callback.
        :param init_types: The names of the module types that should be initialized.
        :param t_max:
        :param std_tol:
        :param verbose:
        """
        self.init_types = tuple(self._str_to_module(m) for m in init_types)
        self.t_max = t_max
        self.std_tol = std_tol
        self.verbose = verbose

    @staticmethod
    def _str_to_module(module_str: str) -> type:
        try:
            return getattr(nn, module_str)
        except AttributeError as err:
            raise ValueError(f"Module {module_str} not found in torch.nn") from err

    @staticmethod
    def _preprocess_batch(module: nn.Module, batch: torch.Tensor | dict) -> dict:
        # Get the function's signature
        sig = inspect.signature(module.forward)

        # Create a set of valid parameter names
        valid_params = set(sig.parameters.keys())
        if isinstance(batch, torch.Tensor) and len(valid_params) == 1:
            (param,) = valid_params
            _batch = {param: batch}
        else:
            _batch = {}
            for k in batch.keys():
                if k in valid_params:
                    _batch.update({k: batch[k]})
        return _batch

    def on_fit_start(self, trainer, pl_module):
        # Get a mini-batch from the train dataloader
        batch = next(iter(trainer.train_dataloader))
        inputs = batch[0] if isinstance(batch, (tuple, list)) else batch
        inputs = move_data_to_device(inputs, device=pl_module.device)

        self.apply_lsuv(pl_module, batch=inputs)

    def apply_lsuv(self, model, batch: dict):
        model.eval()

        def lsuv_hook(module: nn.Module, input, output):
            # TODO: For whatever reason, the input is packed into a tuple by torch
            (input,) = input
            with torch.no_grad():
                nn.init.orthogonal_(module.weight)
                for t in range(self.t_max):
                    output = module.forward(input)
                    output_std = output.std()
                    if abs(output_std - 1) < self.std_tol:
                        break
                    module.weight.data /= output_std + 1e-6
                if self.verbose:
                    logger.debug(f"Layer {module.__class__.__name__}: final std = {output_std:.4f}")
            return output

        # Register the LSUV hook for the entire model
        handles = [m.register_forward_hook(lsuv_hook) for m in model.model.modules() if isinstance(m, self.init_types)]
        batch = self._preprocess_batch(model.model, batch)
        model.model(**batch)

        # Remove the hook after initialization
        for handle in handles:
            handle.remove()

        logger.info("LSUV initialization completed")

from functools import partial

import torch
from loguru import logger
from pytorch_lightning import Callback


class GradientClipping(Callback):
    """
    PyTorch Lightning callback that registers backward hooks for gradient clipping
    during model initialization.
    """

    def __init__(self, clip_value: float):
        """
        Args:
            clip_value (float): Maximum allowed value for gradients. Gradients will be clipped to
                              [-clip_value, clip_value].
        """
        super().__init__()
        self.clip_value = clip_value
        self._hooks = []  # Store hooks to properly remove them if needed

    @staticmethod
    def _clip_gradients(grad, clip_value):
        """Gradient clipping function to be used as a hook."""
        return torch.clamp_(grad, -clip_value, clip_value)

    def on_fit_start(self, trainer, pl_module):
        """Register gradient clipping hooks when training starts."""
        clip_fn = partial(self._clip_gradients, clip_value=self.clip_value)

        # Register hooks for all parameters that require gradients
        for param in pl_module.parameters():
            if param.requires_grad:
                hook = param.register_hook(clip_fn)
                self._hooks.append(hook)

        logger.info(f"Registered gradient clipping hooks with clip_value={self.clip_value}")

    def on_fit_end(self, trainer, pl_module):
        """Remove hooks when training ends to prevent memory leaks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.info("Removed gradient clipping hooks")

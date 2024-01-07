from typing import NamedTuple, Optional
from torch import Tensor


class Targets(NamedTuple):
    """
    A container for the training targets.
    For compatibility with TorchScript, NamedTuple is the best option.
    """

    label: Tensor
    center: Tensor
    wlh: Tensor
    orientation: Tensor
    attribute: Optional[Tensor] = None
    velocity: Optional[Tensor] = None


class TargetHelpers:
    """
    A helper class to facilitate working with the targets
    """

    def __init__(self, targets: Targets):
        """
        Takes an instance of Targets and stores it internally as a dictionary to perform modifications.
        This is intended to be used only internally by the dataset or model, but never to be passed around.
        :param targets:
        """
        self.targets = targets._asdict()

    def __getitem__(self, item) -> Tensor:
        """Returns the specified property of the targets."""
        return self.targets[item]

    def nt(self) -> Targets:
        """Returns the targets as a NamedTuple"""
        return Targets(**self.targets)

import torch
from torchmetrics import Metric


class FeaturemapLevelAssignments(Metric):
    # pylint: disable=no-member
    def __init__(self, num_levels):
        super().__init__()
        self.add_state("level_assignment_count", default=torch.zeros(num_levels).float(), dist_reduce_fx="sum")

    def update(self, count) -> None:  # pylint: disable=arguments-differ
        self.level_assignment_count += count

    def compute(self) -> torch.Tensor:
        return self.level_assignment_count


class PreassignmentOutflows(Metric):
    # pylint: disable=no-member
    def __init__(self):
        super().__init__()
        self.add_state("outflow", default=torch.zeros(1).float(), dist_reduce_fx="sum")

    def update(self, count) -> None:  # pylint: disable=arguments-differ
        self.outflow += count

    def compute(self) -> torch.Tensor:
        return self.outflow


class MissedTargets(Metric):
    # pylint: disable=no-member
    def __init__(self):
        super().__init__()
        self.add_state("missed_targets", default=torch.zeros(1).float(), dist_reduce_fx="sum")

    def update(self, count: int | torch.Tensor) -> None:  # pylint: disable=arguments-differ
        self.missed_targets += count

    def compute(self) -> torch.Tensor:
        return self.missed_targets

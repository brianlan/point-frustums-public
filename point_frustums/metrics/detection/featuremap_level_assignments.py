import torch
from torchmetrics import Metric


class FeaturemapLevelAssignments(Metric):
    # pylint: disable=no-member
    def __init__(self, num_levels):
        super().__init__()
        self.add_state("level_assignment_count", default=torch.zeros(num_levels).int(), dist_reduce_fx="sum")

    def update(self, count) -> None:  # pylint: disable=arguments-differ
        self.level_assignment_count += count

    def compute(self) -> torch.Tensor:
        return self.level_assignment_count

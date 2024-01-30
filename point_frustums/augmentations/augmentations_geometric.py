from collections.abc import MutableMapping
from typing import Optional

import torch

from point_frustums.ops.rotation_conversions import matrix_to_euler_angles, euler_angles_to_matrix
from point_frustums.geometry.utils import angle_to_neg_pi_to_pi
from point_frustums.utils.targets import Targets
from .augmentations import RandomAugmentation


class RandomFlipHorizontal(RandomAugmentation):
    """
    Randomly flip horizontally along the y-axis of the lidar COOS.
    """

    def __init__(self, dimension_mapping_lidar, **kwargs):
        super().__init__(**kwargs)
        self.dimension_mapping_lidar = dimension_mapping_lidar
        self.flip_axis = "y"
        self.flip_index = self.dimension_mapping_lidar[self.flip_axis]

    def _refresh(self):
        # No need to do anything, the data is either flipped or not
        pass

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]) -> torch.Tensor:
        data[:, self.flip_index] *= -1
        azi_idx = self.dimension_mapping_lidar["azimuthal"]
        data[:, azi_idx] = angle_to_neg_pi_to_pi(-data[:, azi_idx])
        return data

    def targets(self, targets: Targets):
        # Flip along the y-axis
        targets.center[:, self.flip_index] *= -1

        # Flip the orientation, this is no physical transformation and therefore requires some extra steps
        #   1. Obtain the euler angles
        targets_euler = matrix_to_euler_angles(targets.orientation, convention="XYZ")
        #   2. Flip the r_x and r_z angles (the rotations corresponding to both axis other than the flipped one)
        assert self.flip_axis == "y", f"It appears as if {self.flip_axis=} has been changed but change was missed here."
        # TODO: Get rid of the euler angles representation and then drop the rotation_conversions.py file
        targets_euler[:, [0, 2]] = -targets_euler[:, [0, 2]]
        #   3. Convert back to the matrix form
        targets.orientation[...] = euler_angles_to_matrix(targets_euler, convention="XYZ")

        # Flip the velocity similarly
        if targets.velocity is not None:
            targets.velocity[:, self.flip_index] *= -1

        return targets

    def metadata(self, metadata: MutableMapping):
        if "velocity" in metadata:
            metadata["velocity"][self.flip_index] *= -1

        return metadata

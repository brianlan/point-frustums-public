from collections.abc import MutableMapping
from math import pi
from random import random
from typing import Optional

import torch

from point_frustums.geometry.quaternion import quaternion_from_rotation_matrix, quaternion_to_rotation_matrix
from point_frustums.geometry.rotation_matrix import rotation_matrix_from_axis_angle
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
        self.flip_along_cart = "y"
        self.flip_spherical = "azimuthal"
        # This corresponds to the channels in the pointcloud and needs not be xyz-first
        self.pc_flip_cart_channel_idx = self.dimension_mapping_lidar[self.flip_along_cart]
        self.pc_flip_sph_channel_idx = self.dimension_mapping_lidar[self.flip_spherical]
        # If data is flipped along one axis, the quaternion components of the other two axis need to be flipped
        self._dims = ("x", "y", "z")
        self.quaternion_flip_indices = [self._dims.index(dim) + 1 for dim in self._dims if dim != self.flip_along_cart]

    def _refresh(self):
        # No need to do anything, the data is either flipped or not
        pass

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]) -> torch.Tensor:
        data[:, self.pc_flip_cart_channel_idx] *= -1
        # Constraining the angle might be required as the azimuth is in the range (-pi, pi]
        data[:, self.pc_flip_sph_channel_idx] = angle_to_neg_pi_to_pi(-data[:, self.pc_flip_sph_channel_idx])
        return data

    def targets(self, targets: Targets):
        # Flip the center along the specified axis
        targets["center"][:, self._dims.index(self.flip_along_cart)] *= -1

        # Mirroring of the orientation along one axis is achieved by flipping the components of the other two axis.
        targets_quaternion = quaternion_from_rotation_matrix(targets["orientation"])
        targets_quaternion[..., self.quaternion_flip_indices] *= -1
        targets["orientation"][...] = quaternion_to_rotation_matrix(targets_quaternion)

        # Flip the velocity similarly
        if targets["velocity"] is not None:
            targets["velocity"][:, self._dims.index(self.flip_along_cart)] *= -1

        return targets

    def metadata(self, metadata: MutableMapping):
        if "velocity" in metadata:
            metadata["velocity"][self._dims.index(self.flip_along_cart)] *= -1

        return metadata


class RandomRotate(RandomAugmentation):
    """
    Randomly rotate the PC about the z-axis.
    """

    def __init__(self, dimension_mapping_lidar, **kwargs):
        super().__init__(**kwargs)
        self.dimension_mapping_lidar = dimension_mapping_lidar
        # This corresponds to the channels in the pointcloud and needs not be xyz-first
        self._azi_idx = self.dimension_mapping_lidar["azimuthal"]
        # If data is flipped along one axis, the quaternion components of the other two axis need to be flipped
        self._dims = ("x", "y", "z")
        self._dims_cart = tuple(self.dimension_mapping_lidar[d] for d in self._dims)

        # We need to generate a random angle on each run
        self._z_axis = torch.tensor([0, 0, 1.0])[None, :]
        self._angle = None
        self._matrix = None

    def _refresh(self):
        # Generate a random rotation angle
        self._angle = torch.tensor(2 * pi * random())
        # Augmentations are generally done by the dataloader on the CPU which is why I don't specify a device here
        self._matrix = rotation_matrix_from_axis_angle(axis=self._z_axis, angle=self._angle)[0, ...]

    def camera(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def radar(self, data: torch.Tensor, metadata: Optional[MutableMapping]):
        raise NotImplementedError()

    def lidar(self, data: torch.Tensor, metadata: Optional[MutableMapping]) -> torch.Tensor:
        data[..., self._dims_cart] = torch.einsum("ij,...j->...i", self._matrix, data[..., self._dims_cart])
        data[..., self._azi_idx] = angle_to_neg_pi_to_pi(data[..., self._azi_idx])
        return data

    def targets(self, targets: Targets):
        # Rotate the center vector
        targets["center"][...] = torch.einsum("ij,...j->...i", self._matrix, targets["center"])
        # Apply the rotation to the orientation matrix
        targets["orientation"][...] = torch.einsum("ij,...jk->...ik", self._matrix, targets["orientation"])
        # Flip the velocity similarly
        if targets["velocity"] is not None:
            targets["velocity"][...] = torch.einsum("ij,...j->...i", self._matrix, targets["velocity"])

        return targets

    def metadata(self, metadata: MutableMapping):
        if "velocity" in metadata:
            metadata["velocity"] = torch.einsum("ij,...j->...i", self._matrix, metadata["velocity"])
        metadata["augmentations"][str(self)] = {"angle": self._angle.item(), "axis": self._z_axis.tolist()}
        return metadata

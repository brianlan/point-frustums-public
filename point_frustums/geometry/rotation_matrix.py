import torch

from .rotation_conversion import (
    _rotation_matrix_to_quaternion,
    _quaternion_to_rotation_matrix,
    _spherical_coordinates_to_rotation_matrix,
    _axis_angle_to_rotation_matrix,
    _rotation_6d_to_matrix,
    _matrix_to_rotation_6d,
)


####################################
# BASIC ROTATiON MATRIX OPERATIONS #
####################################


#################################
# CONVERSION TO ROTATiON MATRIX #
#################################


rotation_matrix_from_quaternion = _quaternion_to_rotation_matrix
rotation_matrix_from_spherical_coordinates = _spherical_coordinates_to_rotation_matrix
rotation_matrix_from_rotation_6d = _rotation_6d_to_matrix


def random_rotation_matrix(n: int, device="cpu") -> torch.Tensor:
    """
    Use the Rodrigues-formula [1] to construct `n` random rotation matrices.
    [1]: Q(d, theta) = I + sin(theta) D + (1-cos(theta))D^2  with D = -d @ levi and D^2 = D @ D
    :param n:
    :param device:
    :return:
    """
    random_axis = torch.nn.functional.normalize(torch.rand((n, 3), device=device), dim=-1)
    random_angles = 2 * torch.pi * torch.rand((n, 1, 1), device=device)
    return _axis_angle_to_rotation_matrix(random_axis, random_angles)


def rx_from_roll(roll: torch.Tensor):
    """
    Construct a rotation matrix from the provided roll in 'X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param roll:
    :return:
    """
    zeros = torch.zeros_like(roll)
    ones = torch.ones_like(roll)
    roll_sin = roll.sin()
    roll_cos = roll.cos()
    matrices = [ones, zeros, zeros, zeros, roll_cos, -roll_sin, zeros, roll_sin, roll_cos]
    return torch.stack(matrices, dim=-1).reshape(roll.shape + (3, 3))


def ry_from_pitch(pitch: torch.Tensor):
    """
    Construct a rotation matrix from the provided pitch in 'X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param pitch:
    :return:
    """
    zeros = torch.zeros_like(pitch)
    ones = torch.ones_like(pitch)
    pitch_sin = pitch.sin()
    pitch_cos = pitch.cos()
    matrices = [pitch_cos, zeros, pitch_sin, zeros, ones, zeros, -pitch_sin, zeros, pitch_cos]
    return torch.stack(matrices, dim=-1).reshape(pitch.shape + (3, 3))


def rz_from_yaw(yaw: torch.Tensor):
    """
    Construct a rotation matrix from the provided yaw in 'X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param yaw:
    :return:
    """
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    yaw_sin = yaw.sin()
    yaw_cos = yaw.cos()
    matrices = [yaw_cos, -yaw_sin, zeros, yaw_sin, yaw_cos, zeros, zeros, zeros, ones]
    return torch.stack(matrices, dim=-1).reshape(yaw.shape + (3, 3))


def rotation_matrix_from_roll_pitch_yaw(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert the roll pitch and yaw provided in 'X-Y-Z` extrinsic rotation (Tait-Bryan) to a rotation matrix.
    :param roll:
    :param pitch:
    :param yaw:
    :return:
    """
    return torch.einsum("...ij,...jk,...kl->...il", rz_from_yaw(yaw), ry_from_pitch(pitch), rx_from_roll(roll))


###################################
# CONVERSION FROM ROTATION MATRIX #
###################################


rotation_matrix_to_quaternion = _rotation_matrix_to_quaternion
rotation_matrix_to_rotation_6d = _matrix_to_rotation_6d

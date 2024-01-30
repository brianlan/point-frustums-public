import torch

from point_frustums.geometry.quaternion import (
    random_quaternions,
    quaternion_allclose,
    quaternion_from_spherical_coordinates,
)
from point_frustums.geometry.rotation_conversion import (
    _rotation_matrix_to_quaternion,
    _quaternion_to_rotation_matrix,
    _rotation_6d_to_matrix,
    _matrix_to_rotation_6d,
    _quaternion_to_axis_angle,
    _axis_angle_to_quaternion,
)
from point_frustums.geometry.rotation_matrix import random_rotation_matrix, rotation_matrix_from_spherical_coordinates

N = 1_000_000
allclose_kwargs = {"atol": 1e-6, "rtol": 1e-6}


def test_conversion_to_and_from_quaternion():
    quaternions = random_quaternions(N)
    quat_rec_from_matrix = _rotation_matrix_to_quaternion(_quaternion_to_rotation_matrix(quaternions))
    assert quaternion_allclose(quat_rec_from_matrix, quaternions, **allclose_kwargs)

    quat_rec_from_axis_angle = _axis_angle_to_quaternion(*_quaternion_to_axis_angle(quaternions))
    assert quaternion_allclose(quat_rec_from_axis_angle, quaternions, **allclose_kwargs)


def test_conversion_to_and_from_matrix():
    matrices = random_rotation_matrix(N)
    mat_rec_from_quaternion = _quaternion_to_rotation_matrix(_rotation_matrix_to_quaternion(matrices))
    assert torch.allclose(mat_rec_from_quaternion, matrices, **allclose_kwargs)

    matrix_rec_from_rot6d = _rotation_6d_to_matrix(_matrix_to_rotation_6d(matrices))
    assert torch.allclose(matrix_rec_from_rot6d, matrices, **allclose_kwargs)


def test_conversion_to_and_from_spherical():
    theta = torch.pi * (torch.rand((N,)) - 0.5)
    phi = 2 * torch.pi * (torch.rand((N,)) - 0.5)

    quaternions = quaternion_from_spherical_coordinates(theta, phi)
    matrices = rotation_matrix_from_spherical_coordinates(theta, phi)

    assert quaternion_allclose(quaternions, _rotation_matrix_to_quaternion(matrices), **allclose_kwargs)
    assert torch.allclose(matrices, _quaternion_to_rotation_matrix(quaternions), **allclose_kwargs)

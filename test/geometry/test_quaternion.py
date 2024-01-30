import torch
from point_frustums.geometry.quaternion import (
    quaternion_from_spherical_coordinates,
    random_quaternions,
    quaternion_to_rotation_matrix,
    quaternion_to_roll_pitch_yaw,
    quaternion_yz_projection_angle,
    quaternion_zx_projection_angle,
    quaternion_xy_projection_angle,
    quaternion_and_axis_to_angle,
)
from point_frustums.geometry.rotation_matrix import (
    rz_from_yaw,
    rotation_matrix_to_quaternion,
    rotation_matrix_from_roll_pitch_yaw,
)


N = 1_000_000


def test_random_quaternion_generator():
    quaternions = random_quaternions(N)
    one = torch.tensor(1.0)
    assert torch.allclose(torch.linalg.vector_norm(quaternions, dim=-1), one)  # pylint: disable=not-callable


def test_quaternion_roll_pitch_yaw_decomposition():
    quaternions = random_quaternions(N)
    matrices = quaternion_to_rotation_matrix(quaternions)
    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(quaternions)
    r_reconstructed = rotation_matrix_from_roll_pitch_yaw(roll, pitch, yaw)
    # TODO: The accuracy here is pretty poor
    assert torch.allclose(matrices, r_reconstructed, atol=1e-4, rtol=1e-4)

    r_z = rz_from_yaw(yaw)
    yaw_quat = rotation_matrix_to_quaternion(r_z)
    roll, pitch, yaw = quaternion_to_roll_pitch_yaw(yaw_quat)
    assert torch.allclose(pitch, torch.zeros_like(pitch))
    assert torch.allclose(roll, torch.zeros_like(roll))
    r_reconstructed = rotation_matrix_from_roll_pitch_yaw(roll, pitch, yaw)
    assert torch.allclose(r_z, r_reconstructed, atol=1e-6, rtol=1e-6)


def test_quaternion_projection_angle():
    theta = torch.pi * (torch.rand((N,)) - 0.5)
    phi = 2 * torch.pi * (torch.rand((N,)) - 0.5)
    zeros = torch.zeros_like(theta)
    # Case 1: Only z-rotation
    quaternions = quaternion_from_spherical_coordinates(zeros, phi)
    assert torch.allclose(phi, quaternion_xy_projection_angle(quaternions))
    assert torch.allclose(zeros, quaternion_zx_projection_angle(quaternions))
    assert torch.allclose(zeros, quaternion_yz_projection_angle(quaternions))

    xaxis = torch.tensor([1.0, 0, 0])
    yaxis = torch.tensor([0, 1.0, 0])
    zaxis = torch.tensor([0, 0, 1.0])

    # Case 2: Mixed rotation
    kwargs = {"atol": 1e-6, "rtol": 1e-6}
    quaternions = quaternion_from_spherical_coordinates(theta, phi)
    assert torch.allclose(
        quaternion_and_axis_to_angle(quaternions, zaxis), quaternion_xy_projection_angle(quaternions), **kwargs
    )
    assert torch.allclose(
        quaternion_and_axis_to_angle(quaternions, yaxis), quaternion_zx_projection_angle(quaternions), **kwargs
    )
    assert torch.allclose(
        quaternion_and_axis_to_angle(quaternions, xaxis), quaternion_yz_projection_angle(quaternions), **kwargs
    )

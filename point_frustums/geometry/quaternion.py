import torch

from .rotation_conversion import (
    _spherical_coordinates_to_quaternion,
    _rotation_matrix_to_quaternion,
    _quaternion_to_rotation_matrix,
    _axis_angle_to_quaternion,
    _quaternion_to_axis_angle,
)
from .utils import angle_to_neg_pi_to_pi


###############################
# BASIC QUATERNION OPERATIONS #
###############################


def invert_quaternion(q: torch.Tensor) -> torch.Tensor:
    q[..., 1:] *= -1
    return q


############################
# CONVERSION TO QUATERNION #
############################


quaternion_from_axis_angle = _axis_angle_to_quaternion


def random_quaternions(n: int, device="cpu") -> torch.Tensor:
    """
    Generate `n` random quaternions.
    :param n:
    :param device:
    :return:
    """
    random_unit_vectors = torch.nn.functional.normalize(torch.rand((n, 3), device=device), dim=-1)
    random_angles = torch.pi * torch.rand((n, 1), device=device) - (torch.pi / 2)
    return quaternion_from_axis_angle(random_unit_vectors, random_angles)


# Define aliases
quaternion_from_spherical_coordinates = _spherical_coordinates_to_quaternion
quaternion_from_rotation_matrix = _rotation_matrix_to_quaternion


##############################
# CONVERSION FROM QUATERNION #
##############################


quaternion_to_rotation_matrix = _quaternion_to_rotation_matrix
quaternion_to_axis_angle = _quaternion_to_axis_angle


def _calculate_twist(q: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    # A reference for the code was: https://stackoverflow.com/a/22401169
    # Some alterations were made:
    #   1. Pre-normalization of the axis
    #   2. Multiplication of the real component with the sign of the scalar product between axis and imaginary component
    #      to eliminate arbitrarily flipped signs of the rotation. Testcase: (1) create random angles (2) create
    #      quaternions from spherical coordinates using angles as phi (3) perform swing-twist-decomposition w.r.t. zaxis
    #      (4) extract angle from twist
    #   3. The resulting angle is shifted back to the range (-pi, pi]
    axis = torch.nn.functional.normalize(axis, dim=-1)
    q_re, q_im = q.split([1, 3], dim=-1)

    # Project q_im onto axis (https://en.wikipedia.org/wiki/Vector_projection)
    # -> Omit division by ||axis||==1
    # -> Extract the sign of the rotation
    projection = torch.einsum("...i,...i->...", q_im, axis)[..., None]
    sign_of_rotation = projection.sign()
    projection = projection * axis

    return torch.nn.functional.normalize(torch.cat((sign_of_rotation * q_re, projection), dim=-1))


def quaternion_to_swing_twist(q: torch.Tensor, axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the Swing Twist decomposition (arXiv:1506.05481) of a quaternion with `q = swing * conj(twist)`
    :param q:
    :param axis:
    :return: tuple[swing, twist] where twist is the component that rotates around the provided axis
    """
    twist = _calculate_twist(q, axis)
    swing = apply_quaternion_to_quaternion(q, quaternion_conj(twist))
    return swing, twist


def _quaternion_to_roll(w: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.arctan2(2.0 * (k * j + w * i), 1.0 - 2.0 * (i * i + j * j))


def quaternion_to_roll(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a {wxyz} quaternion to roll (about x-axis) in body frame and `X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param q:
    :return:
    """
    return _quaternion_to_roll(*q.unbind(dim=-1))


def _quaternion_to_pitch(w: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.arcsin(2.0 * (j * w - k * i))


def quaternion_to_pitch(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a {wxyz} quaternion to pitch (about y-axis) in body frame and `X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param q:
    :return:
    """
    return _quaternion_to_pitch(*q.unbind(dim=-1))


def _quaternion_to_yaw(w: torch.Tensor, i: torch.Tensor, j: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return torch.arctan2(2.0 * (k * w + i * j), -1.0 + 2.0 * (w * w + i * i))


def quaternion_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a {wxyz} quaternion to yaw (about z-axis) in body frame and `X-Y-Z` extrinsic rotation (Tait-Bryan).
    :param q:
    :return:
    """
    return _quaternion_to_yaw(*q.unbind(dim=-1))


def quaternion_to_roll_pitch_yaw(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a {wxyz} quaternion to roll (about x-axis), pitch (about y-axis) and yaw (about z-axis) in body frame and
    `X-Y-Z` extrinsic rotation  (Tait-Bryan).
    :param q:
    :return:
    """
    w, i, j, k = q.unbind(dim=-1)
    roll = _quaternion_to_roll(w, i, j, k)
    pitch = _quaternion_to_pitch(w, i, j, k)
    yaw = _quaternion_to_yaw(w, i, j, k)
    return roll, pitch, yaw


def quaternion_xy_projection_angle(q: torch.Tensor) -> torch.Tensor:
    """
    Use only the `w` and the `k` component of the input quaternion to project onto the xy plane and extract the
    rotation about the z-axis.
    :param q:
    :return:
    """
    q = q * q[..., [0]].sign()
    return 2 * torch.arctan2(q[..., 3], q[..., 0])  # * q[..., 3].sign()


def quaternion_zx_projection_angle(q: torch.Tensor) -> torch.Tensor:
    """
    Use only the `w` and the `j` component of the input quaternion to project onto the zx plane and extract the
    rotation about the y-axis.
    :param q:
    :return:
    """
    return 2 * torch.arctan2(q[..., 2], q[..., 0])  # * q[..., 2].sign()


def quaternion_yz_projection_angle(q: torch.Tensor) -> torch.Tensor:
    """
    Use only the `w` and the `i` component of the input quaternion to project onto the yz plane and extract the
    rotation about the x-axis.
    :param q:
    :return:
    """
    return 2 * torch.arctan2(q[..., 1], q[..., 0])  # * q[..., 1].sign()


def quaternion_and_axis_to_angle(q: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """
    Get the angle from a quaternion and axis.
    This is based on the Swing Twist decomposition but omits unnecessary steps.
    :param q:
    :param axis:
    :return:
    """
    twist = _calculate_twist(q, axis)
    norm_twist_im = torch.linalg.vector_norm(twist[:, 1:], dim=-1)  # pylint: disable=not-callable
    return angle_to_neg_pi_to_pi(2 * torch.arctan2(norm_twist_im, twist[..., 0]))


###############################
# BASIC QUATERNION OPERATIONS #
###############################


def quaternion_conj(q: torch.Tensor) -> torch.Tensor:
    """
    Invert the provided batch of quaternions.
    :param q:
    :return:
    """
    scaling = q.new_tensor([1, -1, -1, -1])
    return q * scaling


@torch.jit.script
def apply_quaternion_to_vector(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the quaternion(s) to the batch of vectors. The computation is either conducted pairwise
    (x.shape[0] == q.shape[0] == N) or by broadcasting q to x (q.shape == [1, 4], x.shape == [N, 3]).
    :param q:
    :param x:
    :return:
    """
    q_re, q_im = q.split([1, 3], dim=-1)
    output = (q_re**2 - q_im.square().sum(dim=-1)[..., None]) * x
    output += 2 * torch.einsum("...j,...j->...", q_im, x)[..., None] * q_im
    output += 2 * q_re * torch.linalg.cross(q_im, x)  # pylint: disable=not-callable
    return output


def apply_quaternion_to_2d_vector(q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    batch_dim = x.shape[:-1]
    zeros = x.new_zeros(batch_dim + (1,))
    x = torch.cat((x, zeros), dim=-1)
    return apply_quaternion_to_vector(q=q, x=x)[..., :2]


@torch.jit.script
def apply_quaternion_to_quaternion(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Apply q1 from the left to q2. The computation is either conducted pairwise (q1.shape == q2.shape == [N, 4]) or by
    broadcasting q1 to q2 (q1.shape == [1, 4], q2.shape == [N, 4]). The algorithm implements equation (1) from [1].

    [1]: https://graphics.stanford.edu/courses/cs348a-17-winter/Papers/quaternion.pdf
    :param q1: The quaternion(s) to apply
    :param q2: The quaternions that shall be transformed
    :return: The transformed quaternions
    """
    assert q1.dim() == q2.dim()
    q1_re, q1_im = q1.split([1, 3], dim=-1)
    q2_re, q2_im = q2.split([1, 3], dim=-1)
    q_out = torch.zeros_like(q2)
    q_out[..., 0] = (q1_re * q2_re).squeeze() - torch.einsum("...i,...i->...", q1_im, q2_im)
    q_out[..., 1:] += q1_re * q2_im + q2_re * q1_im
    q_out[..., 1:] += torch.linalg.cross(q1_im, q2_im, dim=-1)  # pylint: disable=not-callable
    return q_out


@torch.jit.script
def rotate_2d(phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Rotates a sequence of 2D vectors by the angles phi.
    The algorithm is a simplification of quaternion rotation for the 2D case.
    :param phi:
    :param x:
    :return:
    """

    q_0, q_1 = phi.cos(), phi.sin()
    x_0, x_1 = x.unbind(dim=-1)
    return torch.stack([q_0 * x_0 - q_1 * x_1, q_1 * x_0 + q_0 * x_1], dim=-1)


def quaternion_isclose(q1: torch.Tensor, q2: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Test if two quaternion batches are (near to) identical.
    :param q1:
    :param q2:
    :return:
    """
    sign_equal = q1[..., 0].sign() == q2[..., 0].sign()
    q2 = q2.clone()
    q2[~sign_equal] *= -1
    return torch.isclose(q1, q2, **kwargs)


def quaternion_allclose(q1: torch.Tensor, q2: torch.Tensor, **kwargs) -> bool:
    """
    Test if two quaternion batches are (near to) identical.
    :param q1:
    :param q2:
    :return:
    """
    sign_equal = q1[..., 0].sign() == q2[..., 0].sign()
    q2 = q2.clone()
    q2[~sign_equal] *= -1
    return torch.allclose(q1, q2, **kwargs)


def standardize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Ensure that the real part of the quaternion is positive and normalize to unit length.
    :param q:
    :return:
    """
    return torch.nn.functional.normalize(q * q[..., [0]].sign(), dim=-1)


@torch.jit.script
def rotate_3d_spherical(theta: torch.Tensor, phi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Rotates a sequence of 3D vectors by the angles from the spherical coordinate system theta and phi.
    The algorithm is based on the conversion from spherical coordinates to quaternions and then performing
    quaternion rotation.
    :param theta:
    :param phi:
    :param x:
    :return:
    """
    q = quaternion_from_spherical_coordinates(phi=phi, theta=theta)
    return apply_quaternion_to_vector(q=q, x=x)

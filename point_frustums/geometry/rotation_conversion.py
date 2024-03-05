import torch


def levi_civita_symbol(device: str | torch.device = "cpu") -> torch.Tensor:
    """
    Construct the Levi-Civita symbol which can be used to represent the cross product (`a x b = -a . levi . b`)
    among others.
    :param device:
    :return:
    """
    levi = torch.zeros((3, 3, 3), device=device)
    # Set even permutations to 1
    levi[[0, 1, 2], [1, 2, 0], [2, 0, 1]] = 1
    # Set uneven permutations to -1
    levi[[2, 1, 0], [1, 0, 2], [0, 2, 1]] = -1
    return levi


@torch.jit.script
def _rotation_matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion from rotation matrix to quaternion representation.
    Implements algorithm: https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf as also
    used by PyQuaternion.
    Technically, the computational load could be reduced by a factor of 4, as each case is evaluated for all matrices
    but in practice, splitting the input into four and evaluating the cases individually is slower after all.
    :param m: The input matrices [..., 3, 3]
    :return:
    """
    m = m.transpose(-1, -2)
    batch_dim = m.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(m.reshape(batch_dim + (9,)), dim=-1)

    case_outer = torch.lt(m22, 0)
    case_inner_1 = torch.gt(m00, m11)
    case_inner_2 = torch.lt(m00, -m11)

    case_1 = (case_outer * case_inner_1).unsqueeze(-1)
    case_2 = (case_outer * (~case_inner_1)).unsqueeze(-1)
    case_3 = ((~case_outer) * case_inner_2).unsqueeze(-1)
    case_4 = ((~case_outer) * (~case_inner_2)).unsqueeze(-1)

    case_1_t = 1 + m00 - m11 - m22
    case_1_value = torch.stack([m12 - m21, case_1_t, m01 + m10, m20 + m02], dim=-1)
    case_2_t = 1 - m00 + m11 - m22
    case_2_value = torch.stack([m20 - m02, m01 + m10, case_2_t, m12 + m21], dim=-1)
    case_3_t = 1 - m00 - m11 + m22
    case_3_value = torch.stack([m01 - m10, m20 + m02, m12 + m21, case_3_t], dim=-1)
    case_4_t = 1 + m00 + m11 + m22
    case_4_value = torch.stack([case_4_t, m12 - m21, m20 - m02, m01 - m10], dim=-1)

    q = case_1 * case_1_value + case_2 * case_2_value + case_3 * case_3_value + case_4 * case_4_value

    case_1_t = case_1_t.unsqueeze(-1)
    case_2_t = case_2_t.unsqueeze(-1)
    case_3_t = case_3_t.unsqueeze(-1)
    case_4_t = case_4_t.unsqueeze(-1)

    q *= 0.5 / torch.sqrt(case_1 * case_1_t + case_2 * case_2_t + case_3 * case_3_t + case_4 * case_4_t)
    return q


@torch.jit.script
def _spherical_coordinates_to_quaternion(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Construct a batch of quaternions from the given polar and azimuthal angle.
    :param theta: polar or inclination angle (top-down, range [0, pi])
    :param phi: azimuthal angle [-pi, pi]
    :return:
    """
    scalar_phi = (phi / 2).sin()
    scalar_theta = (theta / 2).sin()

    complex_phi = (phi / 2).cos()
    complex_theta = (theta / 2).cos()

    # Set up the output array
    q = torch.stack(
        [
            complex_phi * complex_theta,
            -scalar_phi * scalar_theta,
            complex_phi * scalar_theta,
            scalar_phi * complex_theta,
        ],
        dim=-1,
    )
    return q


@torch.jit.script
def _quaternion_to_rotation_matrix(q: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Batched conversion from quaternion to rotation matrix representation.
    Solution to the eigenvalue problem: `m @ v.vec == q * v * q.conjugate()`.

    :param q: The input quaternions [..., 4]
    :param normalize: Boolean whether to normalize the input (not required if unit quaternions are passed)
    :return:
    """
    batch_dim = q.shape[:-1]
    if normalize:
        q = torch.nn.functional.normalize(q)
    matrix = torch.stack(
        [
            1.0 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2),
            2 * (q[..., 1] * q[..., 2] - q[..., 3] * q[..., 0]),
            2 * (q[..., 1] * q[..., 3] + q[..., 2] * q[..., 0]),
            2 * (q[..., 1] * q[..., 2] + q[..., 3] * q[..., 0]),
            1.0 - 2 * (q[..., 1] ** 2 + q[..., 3] ** 2),
            2 * (q[..., 2] * q[..., 3] - q[..., 1] * q[..., 0]),
            2 * (q[..., 1] * q[..., 3] - q[..., 2] * q[..., 0]),
            2 * (q[..., 2] * q[..., 3] + q[..., 1] * q[..., 0]),
            1.0 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2),
        ],
        dim=-1,
    )
    return matrix.reshape(batch_dim + (3, 3))


@torch.jit.script
def _spherical_coordinates_to_rotation_matrix(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix from the euler angles specified as theta and phi in the ZYX convention.
    :param theta:
    :param phi:
    :return:
    """
    one = torch.ones_like(theta)
    zero = torch.zeros_like(theta)
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    r_z = (cos_phi, -sin_phi, zero, sin_phi, cos_phi, zero, zero, zero, one)
    r_z = torch.stack(r_z, dim=-1).reshape(one.shape + (3, 3))
    r_y = (cos_theta, zero, sin_theta, zero, one, zero, -sin_theta, zero, cos_theta)
    r_y = torch.stack(r_y, dim=-1).reshape(one.shape + (3, 3))
    return torch.einsum("...jk,...kl->...jl", r_z, r_y)


def _quaternion_to_axis_angle(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Decompose a quaternion into its axis-angle representation.

    Note: This conversion has a singularity for `|abs(angle)| < EPS`. In this case, the returned axis may not be
    normalized properly and will be meaningless.
    :param q:
    :return:
    """
    norm = torch.linalg.vector_norm(q[..., 1:4], dim=-1)  # pylint: disable=not-callable
    return torch.nn.functional.normalize(q[..., 1:4], dim=-1), 2 * torch.arctan2(norm, q[..., 0])


def _axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Create a batch of unit quaternions from the provided axis and angle.

    Note: In the no-rotation case where `||axis|| == 0` quaternions have issues with numerical stability. The
    implementation relies on `torch.nn.functional.normalize` to avoid division by zero. The subsequent scaling by
    `sin(angle)` ensures that the magnitude of the imaginary component is scaled to zero.

    :param axis:
    :param angle:
    :return:
    """
    if angle.dim() == axis.dim() - 1:
        angle = angle[..., None]
    angle = angle / 2
    # A slightly more efficient approach might be to only normalize the `axis` where `sin(angle) > EPS`.
    return torch.cat((angle.cos(), angle.sin() * torch.nn.functional.normalize(axis, dim=-1)), dim=-1)


@torch.jit.script
def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.jit.script
def _matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _axis_angle_to_rotation_matrix(axis, angle) -> torch.Tensor:
    """
    Use the Rodrigues-formula [1] to construct `n` random rotation matrices.
    [1]: Q(d, theta) = I + sin(theta) D + (1-cos(theta))D^2  with D = -d @ levi and D^2 = D @ D
    :return:
    """
    dyad = -torch.einsum("ij,jkl->ikl", axis, levi_civita_symbol(device=axis.device))
    dyad_square = torch.einsum("ijk,ikl->ijl", dyad, dyad)
    return torch.eye(3, device=axis.device)[None, ...] + angle.sin() * dyad + (1 - angle.cos()) * dyad_square

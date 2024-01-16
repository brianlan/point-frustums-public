import numpy as np
import torch


def deg_to_rad(deg):
    """
    Uses np.pi but works just fine with torch input
    :param deg:
    :return:
    """
    return (deg / 360) * 2 * np.pi


def rad_to_deg(rad):
    """
    Uses np.pi but works just fine with torch input
    :param rad:
    :return:
    """
    return (rad / (2 * np.pi)) * 360


def cart_to_sph_numpy(points: np.ndarray) -> np.ndarray:
    """
    https://stackoverflow.com/questions/4116658/faster-numpy-cartesian-to-spherical-coordinate-conversion
    """
    coos_sph = np.empty(points.shape, dtype=np.float32)
    xy = points[:, 0] ** 2 + points[:, 1] ** 2  # squared distance to observer in ground projection
    coos_sph[:, 0] = np.sqrt(xy + points[:, 2] ** 2)  # Radial distance
    coos_sph[:, 1] = np.arctan2(np.sqrt(xy), points[:, 2])  # Polar/elevation angle (defined from Z-axis down)
    coos_sph[:, 2] = np.arctan2(points[:, 1], points[:, 0])  # Azimuth angle
    return coos_sph


@torch.jit.script
def cart_to_sph_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Transforms a set of input points in cartesian coordinated to spherical coordinates
    :param points:
    :return: [..., 3] where the last dimension contains (radius, polar and azimuthal angle)
    """
    points_sph = torch.empty(points.shape, dtype=torch.float32, device=points.device)
    xy = points[..., 0] ** 2 + points[..., 1] ** 2  # squared distance to observer in ground projection
    points_sph[..., 0] = torch.sqrt(xy + points[..., 2] ** 2)  # Radial distance
    points_sph[..., 1] = torch.arctan2(torch.sqrt(xy), points[..., 2])  # Polar/elevation angle (from Z-axis down)
    points_sph[..., 2] = torch.arctan2(points[..., 1], points[..., 0])  # Azimuth angle
    return points_sph


@torch.jit.script
def sph_to_cart_torch(points: torch.Tensor) -> torch.Tensor:
    """
    Transforms a set of input points in spherical coordinated to cartesian coordinates
    :param points:[..., 3] where the last dimension contains (radius, polar and azimuthal angle)
    :return: [..., 3] where the last dimension contains (x, y and z)
    """
    points_cart = torch.empty(points.shape, dtype=torch.float32, device=points.device)
    points_cart[..., 0] = points[..., 0] * torch.sin(points[..., 1]) * torch.cos(points[..., 2])
    points_cart[..., 1] = points[..., 0] * torch.sin(points[..., 1]) * torch.sin(points[..., 2])
    points_cart[..., 2] = points[..., 0] * torch.cos(points[..., 1])
    return points_cart


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
    q_0 = q[..., 0]
    q_vec = q[..., 1:]

    output = (q_0**2 - q_vec.square().sum(dim=-1))[..., None] * x
    output += 2 * torch.einsum("ij,ij->i", q_vec, x)[..., None] * q_vec
    output += 2 * q_0[..., None] * torch.linalg.cross(q_vec, x)  # pylint: disable=not-callable

    return output


def rotation_matrix_from_spherical(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
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


@torch.jit.script
def filter_fov_spherical(
    points: torch.Tensor,
    index_mapping: dict[str, int],
    fov_azimuthal: tuple[float, float],
    fov_polar: tuple[float, float],
):
    i_az, i_pol = index_mapping["azimuthal"], index_mapping["polar"]
    mask = (
        (points[:, i_az] >= fov_azimuthal[0])
        & (points[:, i_az] < fov_azimuthal[1])
        & (points[:, i_pol] >= fov_polar[0])
        & (points[:, i_pol] < fov_polar[1])
    )
    return points[mask, :]


def angle_to_neg_pi_to_pi(angle: torch.Tensor):
    """
    Shift all angles to be in the range (-pi, pi].
    :param angle:
    :return:
    """
    # Get the integer multiplier required to shift each angle to (-pi, pi]
    shift = torch.ceil(angle / (2 * torch.pi) - 0.5)
    return angle - shift * 2 * torch.pi


@torch.jit.script
def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
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
def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
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

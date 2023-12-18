from typing import Dict, Tuple
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
def filter_fov_spherical(
    points: torch.Tensor,
    index_mapping: Dict[str, int],
    fov_azimuthal: Tuple[float, float],
    fov_polar: Tuple[float, float],
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

import torch


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
    points_sph[..., 1] = torch.arctan2(torch.sqrt(xy), points[..., 2])  # Polar/inclination angle (from Z-axis down)
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

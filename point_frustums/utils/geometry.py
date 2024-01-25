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
def quaternion_from_spherical(phi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
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
    output += 2 * torch.einsum("ij,ij->i", q_im, x)[..., None] * q_im
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
    q = quaternion_from_spherical(phi=phi, theta=theta)
    return apply_quaternion_to_vector(q=q, x=x)


@torch.jit.script
def quaternion_to_rotation_matrix(q: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Batched conversion from quaternion to rotation matrix representation.
    Solution to the eigenvalue problem: `m @ v.vec == q * v * q.conjugate()`.

    :param q: The input quaternions [..., 4]
    :param normalize: Boolean whether to normalize the input (not required if unit quaternions are passed)
    :return:
    """
    batch_dim = q.shape[:-1]
    if normalize:
        q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)  # pylint: disable=not-callable
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
def rotation_matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
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
def get_corners_3d(centers: torch.Tensor, wlh: torch.Tensor, orientation: torch.Tensor) -> torch.Tensor:
    """
    The corner ordering is/has to be compliant with the specification in pytorch3d.ops.box3d_overlap
    Calculate the 3D box corners. The coordinate system is aligned as:
        x -> forward
        y -> left
        z -> up
    :param centers: Cartesian coordinates of the centers [N, 3]
    :param wlh: Boxes width, length and height [N, 3]
    :param orientation: Orientation matrices [N, 3, 3]
    :return: Cartesian coordinates of the 8 corners [N, 8, 3]
    """
    unit_box_corners = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        device=centers.device,
        dtype=centers.dtype,
    )
    unit_box_corners -= 0.5
    # Scale each box; Take care that sizes are given in the format wlh and the unit box is xyz
    corners = (
        unit_box_corners * wlh.index_select(dim=-1, index=torch.tensor([1, 0, 2], device=centers.device))[..., None, :]
    )
    # Rotate each box individually -> (N,8,3):
    #   - keeps N boxes
    #   - contracts last dimension of rotation matrix with the 3 coordinate values of each point
    #   - produces M new corners in 3D
    # [N, 3, 3] x [N, 8, 3]
    # Shift all boxes to align with their respective center
    return torch.einsum("...jk,...lk->...lj", orientation, corners) + centers[..., None, :]


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


@torch.jit.script
def get_spherical_projection_boundaries(
    centers: torch.Tensor,
    wlh: torch.Tensor,
    orientation: torch.Tensor,
    fov_pol: tuple[float, float],
    fov_azi: tuple[float, float],
    delta_pol: float,
    delta_azi: float,
) -> torch.Tensor:
    # Get 8 corner points of 3D boxes [N, 8, 3]
    corners = get_corners_3d(centers=centers, wlh=wlh, orientation=orientation)
    # Transform cartesian corner coordinates to spherical coordinates
    corners = cart_to_sph_torch(corners)
    # Get the 3D box centers in spherical coordinates
    centers = cart_to_sph_torch(centers)

    # The below is required to rectify targets overlapping the angular ambiguity at +-pi
    # Given a target that has corners with negative and positive azimuthal angle, or speaking in terms of the
    #   featuremap some corners on the very left and some on the very right side.
    # -> Most likely the target does not span almost 360° but rather just overlaps the angular ambiguity.

    # Create a mask that indicates corners corresponding to such targets IFF the center is at least pi/2 from
    #   the center of the featuremap located at 0°.
    mask = ~(centers[:, None, 2].sign() == corners[:, :, 2].sign()) & (centers[:, None, 2].abs() > torch.pi / 2)
    fold_over = torch.nonzero(mask).unbind(-1)
    # Those masked corners are now either on the left or right side of the featuremap and have the opposite sign
    #   of the corresponding center. Now we need to shift them by +- 2 pi so that they exit the featuremap on
    #   the other side but have the same sign as the center angle.
    # This is done by adding SIGN(center) * 2 * pi to the angle:
    #   - If the center is on the left side (has negative angle) then the corners from the right side will be
    #       subtracted 2 pi and thus shifted to the left
    #   - If the center is on the right side (has positive angle) then the corners from the left side will be
    #       added 2 pi and thus shifted to the right
    corners[fold_over[0], fold_over[1], 2] += centers[fold_over[0], 2].sign() * torch.pi * 2
    # TODO: Make sure polar/vertical position is not flipped either

    # Now all corners are shifted s.t. the angular range is not [-pi, pi] but instead [0, 2 * pi] and equivalently
    #   for the polar (not problematic) angle
    corners[:, :, 1] -= fov_pol[0]
    corners[:, :, 2] -= fov_azi[0]

    # Get the minimum and maximum azimuthal/polar coordinate values per 3D box corner (note the dimension reorder)
    min_vals = corners[:, :, [2, 1]].min(dim=1).values
    max_vals = corners[:, :, [2, 1]].max(dim=1).values

    # Stack min/max in a [x, y, x, y] fashion
    boundaries = torch.cat((min_vals, max_vals), dim=1)

    # Compute box coordinates
    boundaries[:, [0, 2]] = boundaries[:, [0, 2]] / delta_azi
    boundaries[:, [1, 3]] = boundaries[:, [1, 3]] / delta_pol
    return boundaries


@torch.jit.script
def iou_vol_3d(boxes1: torch.Tensor, boxes2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D IoU between 2 sets of input boxes defined by 8 corner points. If the input is of shape
    boxes1.shape==boxes2.shape==[N, 8, 3] computes the pairwise IoU of shape [N], else if boxes1.shape==[N, 1, 8, 3] and
    boxes2.shape==[1, M, 8, 3] broadcasts to the binary IoU of shape [N, M].
    :param boxes1:
    :param boxes2:
    :return: The IoU and intersecting volume between the input boxes
    """
    if boxes1.shape == boxes2.shape and boxes1.dim() == 3:
        vol, iou = torch.ops.point_frustums.iou_box3d_pairwise(boxes1, boxes2)
    elif boxes1.size(1) == 1 and boxes2.size(0) == 1 and boxes1.dim() == 4:
        boxes1, boxes2 = boxes1[:, 0, :, :], boxes2[0, :, :, :]
        vol, iou = torch.ops.point_frustums.iou_box3d(boxes1, boxes2)
    else:
        raise ValueError(
            f"Invalid input shapes {boxes1.shape}, {boxes2.shape}; please provide either both as "
            f"[N, 8, 3] or boxes1.shape==[N,1,8,3] and boxes2.shape==[1,M,8,3]."
        )

    return iou, vol

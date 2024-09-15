import torch

from point_frustums.models.backbones.frustum_encoder import decorator_relative_angle, decorator_distance_to_mean


def test_decorator_relative_angle():
    """
    Let's assume a frustum angle of 0.8°. As reference, we'll take the forward position of 0° and from there, frustums
    are distributed evenly, so the adjacent 2 frustums are centered at -0.4° -> [-0.8, 0) and 0.4° -> [0, 0.8). The
    decorator function furthermore scales to the frustum angle, so the returned relative angle will be in [-0.5, 0.5)].
    :return:
    """
    # Test Case 1
    delta_rad = torch.deg2rad(torch.tensor(0.8)).item()
    angles = torch.deg2rad(torch.arange(-0.8, 0.8, step=0.2))
    expected = torch.tensor(
        [
            -0.5,
            -0.25,
            0,
            0.25,
            -0.5,
            -0.25,
            0,
            0.25,
        ]
    )[:, None]
    assert torch.allclose(decorator_relative_angle(angles, delta_rad=delta_rad), expected)
    # Test Case 2
    delta_rad = torch.deg2rad(torch.tensor(1)).item()
    angles = torch.deg2rad(torch.arange(-1.5, 1.5, step=0.5))
    expected = torch.tensor([0, -0.5, 0, -0.5, 0, -0.5])[:, None]
    assert torch.allclose(decorator_relative_angle(angles, delta_rad=delta_rad), expected, atol=1e-7)


def test_decorator_distance_to_mean():
    """
    Let's assume we're working with the radial component for better understanding. And let's ignore the standardization
    for the moment, as this is a quantity that is intrinsic to the real data.
    :return:
    """
    pc_channel = torch.tensor([1.0, 2.0, 3.0, 4.0])
    n_frustums = 2
    i_frustum = torch.tensor([0, 0, 1, 1])
    i_unique, i_inv, counts = i_frustum.unique_consecutive(return_inverse=True, return_counts=True)
    result = decorator_distance_to_mean(
        pc_channel,
        n_frustums=n_frustums,
        counts_padded=counts,
        i_frustum=i_frustum,
        i_unique=i_unique,
        i_inv=i_inv,
        std=1,
    )
    expected_mean = torch.tensor([-0.5, 0.5, -0.5, 0.5])[:, None]
    assert torch.allclose(result, expected_mean), f"Expected {expected_mean}, but got {result}"

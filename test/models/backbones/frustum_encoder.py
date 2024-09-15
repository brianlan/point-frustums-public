import torch

from point_frustums.models.backbones.frustum_encoder import decorator_relative_angle


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

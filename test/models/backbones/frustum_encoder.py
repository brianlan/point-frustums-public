import torch

from point_frustums.config_dataclasses.point_frustums import (
    ConfigDiscretize,
    ConfigVectorize,
    ConfigDecorate,
    ConfigReduce,
    DecoratorFunction,
)
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch, sph_to_cart_torch
from point_frustums.models.backbones import FrustumEncoder
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


class TestFrustumEncoder:
    channels_in = ["x", "y", "z", "intensity", "timestamp", "radial", "polar", "azimuthal"]
    discretize = ConfigDiscretize(n_splits_azi=180, n_splits_pol=40, fov_azi_deg=(-180, 180), fov_pol_deg=(60, 100))
    decorate = ConfigDecorate(
        functions=(
            DecoratorFunction(id="relative_angle", channel="azimuthal", std=1.0),
            DecoratorFunction(id="relative_angle", channel="polar", std=1.0),
            DecoratorFunction(id="distance_to_mean", channel="radial", std=1.0),
        ),
        channels_out=("x", "y", "z", "intensity", "timestamp", "radial", "delta_azi", "delta_pol", "delta_r"),
    )
    vectorize = ConfigVectorize(layers=(16,))
    symmetrize = ("max",)
    reduce = ConfigReduce(layers=(16,))
    frustum_endcoder = FrustumEncoder(
        channels_in=channels_in,
        discretize=discretize,
        decorate=decorate,
        vectorize=vectorize,
        symmetrize=symmetrize,
        reduce=reduce,
    )
    n_points = 100
    batch_size = 1
    n = n_points * batch_size
    pc_cart = (torch.rand((n, 3)) - torch.tensor([0.5, 0.5, 0.2])) * torch.tensor([n, 100, 7])
    pc_misc = torch.rand((n, 2))
    pc_sph = cart_to_sph_torch(pc_cart)
    batch = torch.concatenate((pc_cart, pc_misc, pc_sph), dim=1).split(split_size=n_points, dim=0)

    def test_nan_propagation(self):
        # Take the batch and set one specific point to NaN -> polar: 90.1°, azimuthal: 0.1°
        pc = self.batch[0].clone()
        coords_sph = torch.deg2rad(torch.tensor([0, 90.1, 0.1]))
        coords_sph[0] = 20  # Set the radius
        coords_cart = sph_to_cart_torch(coords_sph)
        pc[0, [0, 1, 2]] = coords_cart
        pc[0, [4, 5]] = torch.nan * torch.tensor([1, 1])
        pc[0, [5, 6, 7]] = coords_sph

        # Evaluate the frustum encoder
        featuremap = self.frustum_endcoder.forward([pc])

        # Now we'd expect to find NaNs in the featuremap entry that corresponds to the specified angular position
        # We specified 40 splits and a polar range of [60°, 100°] and the featuremap goes from top to bottom,
        #   so we'd expect to find NaNs in the 31st cell (index 30).
        # We furthermore specified a discretization of 180 azimuthal splits, the featuremap goes left to right from
        #   negative to positive, so it should be the 91st cell horizontally (index 90).
        nan_mask = torch.zeros((40, 180)).bool()
        nan_mask[30, 90] = True

        # Assure that NaNs were propagated to the expected featuremap region but nowhere else
        assert torch.isnan(featuremap[0, :, nan_mask]).all()
        assert ~torch.isnan(featuremap[0, :, ~nan_mask]).any()

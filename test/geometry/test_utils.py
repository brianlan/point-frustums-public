import torch
from point_frustums.geometry.utils import get_featuremap_projection_boundaries


class TestGetFeaturemapProjectionBoundaries:
    """
    The featuremap has a physical relation to targets around, which is parametrized by the featuremap resolution
    [Nx, Ny] and field of view [delta x, delta y]. Targets are therefore visible on a certain position on the featuremap
    and need to be projected to the latter in order to perform reliable training target assignment. This is done by
    continuously traversing over featuremap index coordinates.
    """

    def test_standard_inputs_projection(self):
        """
        Based on the location to the front, the targets should get projected to the middle of the featuremap.
        :return:
        """
        centers = torch.tensor([[10.0, 0.0, 0.0]], dtype=torch.float32)
        wlh = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        orientation = torch.eye(3).unsqueeze(0)
        fov_pol = (3 / 8 * torch.pi, 5 / 8 * torch.pi)
        fov_azi = (-torch.pi, torch.pi)
        delta_pol = (fov_pol[1] - fov_pol[0]) / 32
        delta_azi = (fov_azi[1] - fov_azi[0]) / 360

        boundaries = get_featuremap_projection_boundaries(
            centers, wlh, orientation, fov_pol, fov_azi, delta_pol, delta_azi
        )

        expected_boundaries = torch.tensor([[177, 14, 183, 18]], dtype=torch.float32)
        assert torch.allclose(boundaries, expected_boundaries, atol=5e-1)

    def test_angular_ambiguity_overlap(self):
        """
        The angular ambiguity is due to the fact that atan2 is in (-pi, pi]. This implies that a target to the back and
        slightly to the right should be projected to the left side of the featuremap with correct x1<x2.
        :return:
        """
        centers = torch.tensor([[-10.0, -0.01, 0.0]], dtype=torch.float32)
        wlh = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
        orientation = torch.eye(3).unsqueeze(0)
        fov_pol = (3 / 8 * torch.pi, 5 / 8 * torch.pi)
        fov_azi = (-torch.pi, torch.pi)
        delta_pol = (fov_pol[1] - fov_pol[0]) / 32
        delta_azi = (fov_azi[1] - fov_azi[0]) / 360

        boundaries = get_featuremap_projection_boundaries(
            centers, wlh, orientation, fov_pol, fov_azi, delta_pol, delta_azi
        )

        # Based on the location to the back but with slightly negative y coordinate, the targets should get projected to
        # the middle of the featuremap
        expected_boundaries = torch.tensor([[-3, 14, 3, 18]], dtype=torch.float32)
        assert torch.allclose(boundaries, expected_boundaries, atol=5e-1)

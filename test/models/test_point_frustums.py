from unittest.mock import Mock
import torch
from point_frustums.models.point_frustums import PointFrustums
from point_frustums.geometry.rotation_matrix import random_rotation_matrix
from point_frustums.geometry.coordinate_system_conversion import cart_to_sph_torch

# pylint: disable=protected-access


class TestPointFrustumsCoders:
    n = 1000
    module_mock = Mock()
    module_mock.feature_vectors_angular_centers = torch.arange(0, 2 * n).reshape((n, 2)) / n
    module_mock.discretization.delta_pol = 0.01
    module_mock.discretization.delta_azi = 0.01

    def test_center_coder(self):
        points_cartesian: torch.Tensor = torch.rand((self.n, 3))
        points_spherical = cart_to_sph_torch(points_cartesian)
        points_encoded = PointFrustums._encode_center(self.module_mock, points_spherical)
        points_decoded = PointFrustums._decode_center(self.module_mock, points_encoded)
        assert torch.allclose(points_spherical, points_decoded, atol=1e-6)

    def test_velocity_coder(self):
        ego_velocity = torch.rand(1, 3).repeat((self.n, 1))
        velocities_cartesian: torch.Tensor = torch.rand((self.n, 3))
        velocities_encoded = PointFrustums._encode_velocity(self.module_mock, velocities_cartesian, ego_velocity)
        velocities_decoded = PointFrustums._decode_velocity(self.module_mock, velocities_encoded, ego_velocity)
        assert torch.allclose(velocities_cartesian[:, [0, 1]], velocities_decoded, atol=1e-6)

    def test_orientation_coder(self):
        rotation_matrices = random_rotation_matrix(self.n)
        orientation_encoded = PointFrustums._encode_orientation(self.module_mock, rotation_matrices)
        orientation_decoded = PointFrustums._decode_orientation(self.module_mock, orientation_encoded)
        assert torch.allclose(rotation_matrices, orientation_decoded, atol=1e-6)

    def test_wlh_coder(self):
        wlh = torch.rand((self.n, 3))
        wlh_encoded = PointFrustums._encode_wlh(wlh)
        wlh_decoded = PointFrustums._decode_wlh(wlh_encoded)
        assert torch.allclose(wlh, wlh_decoded, atol=1e-6)

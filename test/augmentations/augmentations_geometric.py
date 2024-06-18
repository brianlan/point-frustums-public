from copy import deepcopy as deepcp
import torch
from point_frustums.geometry.rotation_matrix import rotation_matrix_from_axis_angle
from point_frustums.augmentations.augmentations_geometric import RandomRotate, RandomFlipHorizontal

# pylint: disable=protected-access

# Create a sample point (x, y, z, intensity, timestamp, radial, polar, azimuthal) cloud with a few selected points.
PC = {
    "LIDAR_TOP": torch.tensor(
        [
            [50, 0, 0, 255, 0, 50, torch.pi / 2, 0],  # forward
            [0, 50, 0, 255, 0, 50, torch.pi / 2, torch.pi / 2],  # left
            [0, -50, 0, 255, 0, 50, torch.pi / 2, -torch.pi / 2],  # right
            [-50, 1e-8, 0, 255, 0, 50, torch.pi / 2, torch.pi],  # backward left
            [-50, -1e-8, 0, 255, 0, 50, torch.pi / 2, -torch.pi + 1e-6],  # backward right
        ],
        dtype=torch.float32,
    )
}
TARGETS = {
    "class": None,
    "attribute": None,
    "center": torch.tensor(
        [
            [15, 0, 0],
            [0, 10, 0],
            [20, 0, 0],
            [10, -1, 0],
        ],
        dtype=torch.float32,
    ),
    "wlh": None,
    "orientation": rotation_matrix_from_axis_angle(
        axis=torch.tensor([0, 0, 1], dtype=torch.float32)[None, :],
        angle=torch.tensor([0, 0, torch.pi / 2, torch.pi], dtype=torch.float32),
    ),
    "velocity": torch.tensor(
        [
            [3, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [-5, 0, 0],
        ],
        dtype=torch.float32,
    ),
}
METADATA = {"velocity": torch.tensor([5, 0, 0], dtype=torch.float32)}
AUGMENTAION_KWARGS = {
    "dimension_mapping_lidar": {
        "x": 0,
        "y": 1,
        "z": 2,
        "intensity": 3,
        "timestamp": 4,
        "radial": 5,
        "polar": 6,
        "azimuthal": 7,
    },
    "probability": 1.0,
    "apply_to": {"lidar", "targets"},
}


class TestRandomRotate:
    augmentation = RandomRotate(**AUGMENTAION_KWARGS)

    def test_no_rotation(self):
        """Everything should remain unchanged"""
        self.augmentation._angle = torch.tensor(0)
        self.augmentation._matrix = rotation_matrix_from_axis_angle(
            axis=self.augmentation._z_axis,
            angle=self.augmentation._angle,
        )[0, ...]
        lidar, _, _, targets, metadata = self.augmentation(deepcp(PC), None, None, deepcp(TARGETS), deepcp(METADATA))
        assert torch.allclose(lidar["LIDAR_TOP"], PC["LIDAR_TOP"])
        assert torch.allclose(targets["center"], TARGETS["center"])
        assert torch.allclose(targets["orientation"], TARGETS["orientation"])
        assert torch.allclose(targets["velocity"], TARGETS["velocity"])
        assert torch.allclose(metadata["velocity"], METADATA["velocity"])

    def test_rotation_by_pi_half(self):
        self.augmentation._angle = torch.tensor(torch.pi / 2)
        self.augmentation._matrix = rotation_matrix_from_axis_angle(
            axis=self.augmentation._z_axis, angle=self.augmentation._angle
        )[0, ...]
        lidar, _, _, targets, metadata = self.augmentation(deepcp(PC), None, None, deepcp(TARGETS), deepcp(METADATA))

        pc_transformed = torch.tensor(
            [
                [0, 50, 0, 255, 0, 50, torch.pi / 2, torch.pi / 2],  # left
                [-50, 1e-8, 0, 255, 0, 50, torch.pi / 2, torch.pi],  # backward left
                [50, 0, 0, 255, 0, 50, torch.pi / 2, 0],  # forward
                [0, -50, 0, 255, 0, 50, torch.pi / 2, -torch.pi / 2],  # right
                [0, -50, 0, 255, 0, 50, torch.pi / 2, -torch.pi / 2],  # right
            ],
            dtype=torch.float32,
        )
        targets_transformed = {
            "class": None,
            "attribute": None,
            "center": torch.tensor(
                [
                    [0, 15, 0],
                    [-10, 0, 0],
                    [0, 20, 0],
                    [1, 10, 0],
                ],
                dtype=torch.float32,
            ),
            "wlh": None,
            "orientation": rotation_matrix_from_axis_angle(
                axis=torch.tensor([0, 0, 1], dtype=torch.float32)[None, :],
                angle=torch.tensor([0, 0, torch.pi / 2, torch.pi], dtype=torch.float32) + self.augmentation._angle,
            ),
            "velocity": torch.tensor(
                [
                    [0, 3, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -5, 0],
                ],
                dtype=torch.float32,
            ),
        }

        assert torch.allclose(lidar["LIDAR_TOP"], pc_transformed, atol=1e-4)
        assert torch.allclose(targets["center"], targets_transformed["center"])
        assert torch.allclose(targets["orientation"], targets_transformed["orientation"], atol=1e-5)
        assert torch.allclose(targets["velocity"], targets_transformed["velocity"])
        assert torch.allclose(metadata["velocity"], torch.tensor([0, 5, 0], dtype=torch.float32))

    def test_rotation_by_minus_pi_half(self):
        self.augmentation._angle = torch.tensor(-torch.pi / 2)
        self.augmentation._matrix = rotation_matrix_from_axis_angle(
            axis=self.augmentation._z_axis, angle=self.augmentation._angle
        )[0, ...]
        lidar, _, _, targets, metadata = self.augmentation(deepcp(PC), None, None, deepcp(TARGETS), deepcp(METADATA))
        # forward, left, right, backward left, backward right -> right, forward, backward, left, left
        pc_transformed = torch.tensor(
            [
                [0, -50, 0, 255, 0, 50, torch.pi / 2, -torch.pi / 2],  # right
                [50, 0, 0, 255, 0, 50, torch.pi / 2, 0],  # forward
                [-50, 1e-8, 0, 255, 0, 50, torch.pi / 2, torch.pi],  # backward left
                [0, 50, 0, 255, 0, 50, torch.pi / 2, torch.pi / 2],  # left
                [0, 50, 0, 255, 0, 50, torch.pi / 2, torch.pi / 2],  # left
            ],
            dtype=torch.float32,
        )
        targets_transformed = {
            "class": None,
            "attribute": None,
            "center": torch.tensor(
                [
                    [0, -15, 0],
                    [10, 0, 0],
                    [0, -20, 0],
                    [-1, -10, 0],
                ],
                dtype=torch.float32,
            ),
            "wlh": None,
            "orientation": rotation_matrix_from_axis_angle(
                axis=torch.tensor([0, 0, 1], dtype=torch.float32)[None, :],
                angle=torch.tensor([0, 0, torch.pi / 2, torch.pi], dtype=torch.float32) + self.augmentation._angle,
            ),
            "velocity": torch.tensor(
                [
                    [0, -3, 0],
                    [0, -1, 0],
                    [1, 0, 0],
                    [0, 5, 0],
                ],
                dtype=torch.float32,
            ),
        }

        assert torch.allclose(lidar["LIDAR_TOP"], pc_transformed, atol=1e-4)
        assert torch.allclose(targets["center"], targets_transformed["center"])
        assert torch.allclose(targets["orientation"], targets_transformed["orientation"], atol=1e-5)
        assert torch.allclose(targets["velocity"], targets_transformed["velocity"])
        assert torch.allclose(metadata["velocity"], torch.tensor([0, -5, 0], dtype=torch.float32))


class TestRandomFlipHorizontal:
    augmentation = RandomFlipHorizontal(**AUGMENTAION_KWARGS)

    def test_no_flip(self):
        """Everything should remain unchanged"""
        self.augmentation.apply = False
        lidar, _, _, targets, metadata = self.augmentation(deepcp(PC), None, None, deepcp(TARGETS), deepcp(METADATA))
        assert torch.allclose(lidar["LIDAR_TOP"], PC["LIDAR_TOP"])
        assert torch.allclose(targets["center"], TARGETS["center"])
        assert torch.allclose(targets["orientation"], TARGETS["orientation"])
        assert torch.allclose(targets["velocity"], TARGETS["velocity"])
        assert torch.allclose(metadata["velocity"], METADATA["velocity"])

    def test_flip_1(self):
        self.augmentation.apply = True
        lidar, _, _, targets, metadata = self.augmentation(deepcp(PC), None, None, deepcp(TARGETS), deepcp(METADATA))

        # forward, left, right, backward left, backward right -> right, forward, backward, left, left
        pc_transformed = torch.tensor(
            [
                [50, 0, 0, 255, 0, 50, torch.pi / 2, 0],  # forward
                [0, -50, 0, 255, 0, 50, torch.pi / 2, -torch.pi / 2],  # right
                [0, 50, 0, 255, 0, 50, torch.pi / 2, torch.pi / 2],  # left
                [-50, -1e-8, 0, 255, 0, 50, torch.pi / 2, torch.pi - 1e-6],  # backward right
                [-50, 1e-8, 0, 255, 0, 50, torch.pi / 2, torch.pi],  # backward left
            ],
            dtype=torch.float32,
        )
        targets_transformed = {
            "class": None,
            "attribute": None,
            "center": torch.tensor(
                [
                    [15, 0, 0],
                    [0, -10, 0],
                    [20, 0, 0],
                    [10, 1, 0],
                ],
                dtype=torch.float32,
            ),
            "wlh": None,
            "orientation": rotation_matrix_from_axis_angle(
                axis=torch.tensor([0, 0, 1], dtype=torch.float32)[None, :],
                angle=torch.tensor([0, 0, -torch.pi / 2, -torch.pi], dtype=torch.float32),
            ),
            "velocity": torch.tensor(
                [
                    [3, 0, 0],
                    [1, 0, 0],
                    [0, -1, 0],
                    [-5, 0, 0],
                ],
                dtype=torch.float32,
            ),
        }
        assert torch.allclose(lidar["LIDAR_TOP"], pc_transformed, atol=1e-4)
        assert torch.allclose(targets["center"], targets_transformed["center"])
        assert torch.allclose(targets["orientation"], targets_transformed["orientation"], atol=1e-5)
        assert torch.allclose(targets["velocity"], targets_transformed["velocity"])
        assert torch.allclose(metadata["velocity"], torch.tensor([5, 0, 0], dtype=torch.float32))

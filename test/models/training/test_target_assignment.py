import torch
from point_frustums.models.training.target_assignment import match_target_projection_to_receptive_field


class TestMatchTargetProjectionToReceptiveField:
    def test_correct_binary_mask(self):
        targets_projections = torch.tensor([[1, 1, 3, 3], [4, 4, 6, 6]], dtype=torch.float32)
        rf_centers = torch.tensor([[2, 2], [5, 5]], dtype=torch.float32)
        rf_sizes = torch.tensor([[2, 2], [2, 2]], dtype=torch.float32)
        layer_sizes_flat = [1, 1]
        base_featuremap_width = 360

        expected_mask = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
        result_mask = match_target_projection_to_receptive_field(
            targets_projections, rf_centers, rf_sizes, layer_sizes_flat, base_featuremap_width
        )

        assert torch.equal(result_mask, expected_mask)

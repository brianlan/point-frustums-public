from collections.abc import Sequence
from typing import Any, Optional

import torch
from torchmetrics import Metric

from point_frustums.config_dataclasses.dataset import Annotations
from ..functional.nds import (
    _nds_update_distance_function,
    _nds_update_class_match_function,
    _nds_update_target_count,
    _calc_tp_err_attribute,
    _calc_tp_err_velocity,
    _calc_tp_err_orientation,
    _calc_tp_err_scale,
    _calc_tp_err_translation,
)


class NuScenesDetectionScore(Metric):
    tp_metrics_specification = {
        "translation": {"err_fn": _calc_tp_err_translation, "attribute": "center", "id": "ATE"},
        "scale": {"err_fn": _calc_tp_err_scale, "attribute": "wlh", "id": "ASE"},
        "orientation": {"err_fn": _calc_tp_err_orientation, "attribute": "orientation", "id": "AOE"},
        "velocity": {"err_fn": _calc_tp_err_velocity, "attribute": "velocity", "id": "AVE"},
        "attribute": {"err_fn": _calc_tp_err_attribute, "attribute": "attribute", "id": "AAE"},
    }

    def __init__(
        self,
        annotations: Annotations,
        distance_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
        tp_threshold: float = 2.0,
        n_classes: int = 10,
        n_points_interpolation: int = 101,
        min_recall: float = 0.1,
        min_precision: float = 0.1,
        nds_map_weight: float = 5.0,
    ):
        assert tp_threshold in distance_thresholds
        super().__init__()
        # TODO: Steps involved in calculating the NDS
        # TODO: Problem: How to track samples? -> Maybe store dataloader indices
        # TODO: Detection Config
        #   - Filter by {range, bike-rack, n_points(targets) == 0}
        #   - max_boxes per sample
        #   - Compute orientation error for traffic cone only up to 180 deg
        self.annotations = annotations
        self.distance_thresholds = distance_thresholds
        self.tp_threshold = tp_threshold
        self.n_classes = n_classes
        self.n_points_interpolation = n_points_interpolation
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.nds_map_weight = nds_map_weight

        for i_cls in range(self.n_classes):
            # Register the target count state for each class
            self.add_state(f"n_targets_{i_cls}", default=torch.tensor(0), dist_reduce_fx="sum")

        for threshold, _ in enumerate(self.distance_thresholds):
            # Register states applied to all parsed detections
            self.add_state(f"tp_score_t{threshold}", default=[], dist_reduce_fx=None)
            self.add_state(f"tp_class_t{threshold}", default=[], dist_reduce_fx=None)
            self.add_state(f"fp_score_t{threshold}", default=[], dist_reduce_fx=None)
            self.add_state(f"fp_class_t{threshold}", default=[], dist_reduce_fx=None)
            # Register states applied to TP detections (error metrics)
            for metric in self.tp_metrics_specification:
                self.add_state(f"tp_err_{metric}_t{threshold}", default=[], dist_reduce_fx=None)

    def n_targets(self, i: int) -> torch.Tensor:
        return getattr(self, f"n_targets_{i}")

    def _append_list_state(self, reference: str, threshold: int, data: torch.Tensor):
        getattr(self, f"{reference}_t{threshold}").append(data)

    def _increment_target_counts(self, cls_index: list[int], cls_count: list[int]):
        for i, c in zip(cls_index, cls_count):
            n_targets = getattr(self, f"n_targets_{i}")
            n_targets += c

    def update(  # pylint: disable=arguments-differ
        self,
        batch_detections: list[dict[str, torch.Tensor]],
        batch_targets: list[dict[str, torch.Tensor]],
    ):
        """
        Evaluate TP and FP detections for each distance threshold and append to the respective state. Increment the
        target count per class.
        :param batch_detections:
        :param batch_targets:
        :return:
        """
        # Iterate over the samples
        for detections, targets in zip(batch_detections, batch_targets):
            # Store the number of targets per class
            self._increment_target_counts(*_nds_update_target_count(targets["class"]))

            # Evaluate the distance between all detections and all targets and the respective class matches
            distance = _nds_update_distance_function(detections=detections["center"], targets=targets["center"])
            class_match = _nds_update_class_match_function(detections=detections["class"], targets=targets["class"])

            # Iterate over the distance thresholds
            for i, threshold in enumerate(self.distance_thresholds):
                # TODO
                raise NotImplementedError

    def compute(self, output_file: Optional[str] = None) -> Any:
        raise NotImplementedError

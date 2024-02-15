from collections.abc import Iterable, Sequence
from itertools import product
from math import inf, nan, isnan

import numpy as np
import torch

from point_frustums.functional import interpolate_to_support
from point_frustums.geometry.rotation_matrix import rotation_matrix_to_yaw
from point_frustums.geometry.utils import angle_to_neg_pi_to_pi


def cummean(x: torch.Tensor, dim: int = -1, replace_nan: bool = True):
    """
    This replicates the cummean behaviour of the nuscenes-devkit.
    :param x:
    :param dim:
    :param replace_nan:
    :return:
    """
    x_nan = torch.isnan(x)
    if x_nan.all().item():
        return torch.ones_like(x)
    div = (~x_nan).cumsum(dim=dim).clamp(min=1e-8)
    if replace_nan:
        x = x.nan_to_num()
    return x.cumsum(dim=dim) / div


def _calc_tp_err_translation(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    The NDS uses the XY center distance.
    :param detections:
    :param targets:
    :return:
    """
    return torch.linalg.vector_norm(detections[..., :2] - targets[..., :2], dim=-1)  # pylint: disable=not-callable


def _calc_tp_err_scale(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    The NDS uses `1 - 3D IoU` where the IoU is calculated between aligned (w.r.t. translation and rotation)
    detections and targets.
    :param detections:
    :param targets:
    :return:
    """
    intersection = torch.min(detections, targets).prod(dim=-1)
    union = targets.prod(dim=-1) + detections.prod(dim=-1) - intersection
    return 1 - intersection / union


def _calc_tp_err_orientation(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    The NDS extracts the yaw from detections and targets and calculates the difference.
    :param detections: The orientation of the detections in form of rotation matrices
    :param targets: The target orientation in form of rotation matrices
    :return:
    """
    detections_yaw = rotation_matrix_to_yaw(detections)
    targets_yaw = rotation_matrix_to_yaw(targets)
    return angle_to_neg_pi_to_pi(detections_yaw - targets_yaw).abs()


def _calc_tp_err_velocity(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    NuScenes just uses the L2 distance on the xy-plane and so do I.
    :param detections:
    :param targets:
    :return:
    """
    return torch.linalg.vector_norm(detections[..., :2] - targets[..., :2], dim=-1)  # pylint: disable=not-callable


def _calc_tp_err_attribute(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    NuScenes compares the detection and target attribute label here. This method directly compares the integer label
    which is slightly different as it omits the conversion from plain attribute to category.attribute which
    introduces some ambiguity because 2 different categories can have the same attribute.
    Furthermore, NuScenes filters the detections matched to void targets.
    :param detections:
    :param targets:
    :return:
    """
    # TODO: Set void targets to NaN to exclude from the score
    # TODO: Resolve ambiguity before comparing
    return 1 - torch.eq(detections, targets).float()


def _nds_update_target_count(targets: torch.Tensor) -> tuple[list[int], list[int]]:
    """
    Get the number of targets per class in the provided sample.
    :param targets:
    :return: tuple[cls_index, cls_count]
    """
    cls_index, cls_count = targets.unique(return_counts=True)
    return cls_index.tolist(), cls_count.tolist()


def _nds_update_distance_function(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    distance = detections[:, None, ..., :2] - targets[None, ..., :2]
    return torch.linalg.vector_norm(distance, dim=-1)  # pylint: disable=not-callable


def _nds_update_class_match_function(detections: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.eq(detections[:, None, ...], targets[None, ...])


def _resolve_ambiguous_matches(matches_distances: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
    """
    This implements a strategy to resolve ambiguities where a target is matched by more than one detection.
    The NDS permits only one TP per target and therefore iteratively assigns the closest of the remaining targets.
    This imitates the behavior by setting the distances between one target and all other detections to inf.
    :param matches_distances:
    :param score:
    :return:
    """
    score_sort_idx = torch.sort(score, stable=True, descending=True).indices
    matches_distances = matches_distances[score_sort_idx, :]
    for i in range(matches_distances.shape[0]):
        distance, index = matches_distances[i, :].min(dim=0)
        if distance.item() == inf:
            continue
        matches_distances[:, index] = torch.inf
        matches_distances[i, index] = distance

    return matches_distances[score_sort_idx.argsort(), :]


def _nds_update_assign_target(
    threshold: float, distance: torch.Tensor, class_match: torch.Tensor, score
) -> tuple[torch.Tensor, torch.Tensor]:
    matches = torch.lt(distance, threshold) & class_match
    matches_distances = distance.clone()
    matches_distances[~matches] = torch.inf

    # NuScenes assigns iteratively (by descending score) the closest remaining target to each detection.
    # No target is assigned more than once. This is tricky to solve without iterating over detections.
    matches_distances = _resolve_ambiguous_matches(matches_distances, score)
    distance, index = matches_distances.min(dim=-1)
    return distance, index


def _nds_compute_merge_tp_and_fp(
    tp_class: torch.Tensor, tp_score: torch.Tensor, fp_class: torch.Tensor, fp_score: torch.Tensor, i_cls: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    From the provided tp and fp arrays, compute the cumulative sum representation that is used to calculate precision
    and recall. Get also the corresponding merged score (TP + FP) and the index that retrieves the sorted subset for
    the class index.
    :param tp_class:
    :param tp_score:
    :param fp_class:
    :param fp_score:
    :param i_cls:
    :return:
    """
    mask_tp_subset_cls = torch.eq(tp_class, i_cls)
    mask_fp_subset_cls = torch.eq(fp_class, i_cls)

    # Subset all states to the class
    tp_score_subset = tp_score[mask_tp_subset_cls]
    fp_score_subset = fp_score[mask_fp_subset_cls]

    # Sort all TP states by decreasing score and construct the subset and sort index
    tp_score_subset, tp_sort_index = tp_score_subset.sort(descending=True)
    tp_subset_sort_index = mask_tp_subset_cls.nonzero().squeeze(1)[tp_sort_index]

    # Merge TP and FP score and get the descending-sorted version together with the sort index
    merged_score, merged_sort_idx = torch.cat((tp_score_subset, fp_score_subset), dim=0).sort(descending=True)
    # Create, sort and accumulate TP and FP count
    tp = torch.cat((torch.ones_like(tp_score_subset), torch.zeros_like(fp_score_subset)))[merged_sort_idx].cumsum(dim=0)
    fp = torch.cat((torch.zeros_like(tp_score_subset), torch.ones_like(fp_score_subset)))[merged_sort_idx].cumsum(dim=0)
    return tp, fp, merged_score, tp_subset_sort_index


def _nds_compute_interpolate_recall_precision_score(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    n_targets: torch.Tensor,
    score: torch.Tensor,
    n_points: int,
):
    """
    Interpolate precision and score to the recall in the range [0, 1].
    :param true_positive:
    :param false_positive:
    :param n_targets:
    :param score:
    :param n_points:
    :return:
    """
    recall_support = torch.linspace(0, 1, n_points, device=true_positive.device)

    # Calculate precision and recall w.r.t. the number of predictions and seen targets
    precision = true_positive / (false_positive + true_positive)
    recall = true_positive / n_targets.clamp(min=1e-8)

    # Interpolate precision and recall for the specified number of recall points (default=101)
    interpolated_metrics = {
        "recall": recall_support,
        "precision": interpolate_to_support(x=recall, y=precision, support=recall_support, right=0),
        "score": interpolate_to_support(x=recall, y=score, support=recall_support, right=0),
    }
    return interpolated_metrics


def _nds_compute_calculate_ap(precision: torch.Tensor, min_recall: float, min_precision: float, n_bins: int) -> float:
    """
    Calculate the average precision as done by NuScenes. The paper states: '...the area under the precision recall
    curve  for recall and precision over 10%. Operating points where recall or precision is less than 10% are
    removed...'
    :param precision: The interpolated precision.
    :return:
    """
    min_recall_bin_index = round(min_recall * (n_bins - 1))
    normalization_factor = 1 - min_precision
    return (precision[min_recall_bin_index + 1 :] - min_precision).clamp(min=0).mean() / normalization_factor


def _nds_compute_interpolate_tp_error(score, error, score_support):
    """
    Interpolate the error to the score in the range [0, 1]. Uses the cumulative mean of the error.
    :param score:
    :param error:
    :param score_support:
    :return:
    """

    if error.numel() == 0:
        return torch.ones_like(score_support)

    return interpolate_to_support(x=score, y=cummean(error), support=score_support, order="decreasing")


def _nds_compute_calculate_tp_error(
    error: torch.Tensor,
    score: torch.Tensor,
    metric_name: str,
    class_name: str,
    min_recall: float,
    n_bins: int,
) -> float:
    """
    Apply the min_recall threshold and compute the average error.
    :param error:
    :param score:
    :param metric_name:
    :param class_name:
    :param min_recall:
    :param n_bins:
    :return:
    """
    if class_name in ("traffic_cone",) and metric_name in ("orientation", "velocity", "attribute"):
        return nan
    if class_name in ("barrier",) and metric_name in ("attribute", "velocity"):
        return nan

    # Everything below min_recall shall be excluded from the calculation
    min_recall_bin_index_expected = round(min_recall * (n_bins - 1)) + 1
    # The highest recall measured with nonzero score
    max_recall_bin_index_actual = torch.gt(score, 0).count_nonzero().item()

    # None of the detections passed the 0.1 threshold
    if max_recall_bin_index_actual - 1 < min_recall_bin_index_expected:
        return 1.0

    return error[min_recall_bin_index_expected:max_recall_bin_index_actual].mean().item()


def _nds_compute_average_ap_over_thresholds(
    recall_averaged_metrics: dict[int, dict[int, dict[str, float]]],
    thresholds: Sequence[float],
    n_classes: int,
) -> dict[int, float]:
    """
    Returns a mapping from class index to the class AP.
    :param recall_averaged_metrics:
    :param thresholds:
    :param n_classes:
    :return:
    """
    average_ap = {i: [] for i in range(n_classes)}
    for i_thresh, i_cls in product(range(len(thresholds)), range(n_classes)):
        ap = recall_averaged_metrics[i_thresh][i_cls]["AP"]
        if not isnan(ap):
            average_ap[i_cls].append(ap)
    return {i_cls: float(np.mean(v)) for i_cls, v in average_ap.items()}


def _nds_compute_select_tp_metric_from_thresholds(
    recall_averaged_metrics: dict[int, dict[int, dict[str, float]]],
    threshold_idx: int,
    n_classes: int,
    tp_metrics: Sequence[str],
) -> dict[str, dict[int, float]]:
    """
    Returns a nested mapping from the metric name to the class index to the class TP error.
    :param recall_averaged_metrics:
    :param threshold_idx:
    :param n_classes:
    :param tp_metrics:
    :return:
    """
    metrics = {m: {} for m in tp_metrics}
    for metric, i_cls in product(tp_metrics, range(n_classes)):
        metrics[metric][i_cls] = float(recall_averaged_metrics[threshold_idx][i_cls][metric])
    return metrics


def _filter_nan(values: Iterable) -> list:
    """
    Return a subset of the iterable that does not contain any NaN values.
    :param values:
    :return:
    """
    return [v for v in values if not isnan(v)]


def _nds_compute_mean_metrics(class_metrics: dict[str, dict[int, float]]) -> dict[str, float]:
    """
    Average the per-class error to get the mean TP error. Returns a mapping containing all mTP errors.
    :param class_metrics:
    :return:
    """
    return {metric: float(np.mean(_filter_nan(class_vals.values()))) for metric, class_vals in class_metrics.items()}


def _nds_compute_mean_ap(class_ap: dict[int, float]) -> float:
    """
    Average the per-class average precision to get the mean average precision.
    :param class_ap:
    :return:
    """
    return float(np.mean(list(class_ap.values())))


def _nds_compute_nds(averaged_ap: float, averaged_metrics: dict[str, float], ap_weight: float = 5.0) -> float:
    scores = {metric: max(0.0, 1 - value) for metric, value in averaged_metrics.items()}
    return float((ap_weight * averaged_ap + np.sum(tuple(scores.values()))) / (ap_weight + len(scores)))

from math import inf

import torch
from point_frustums.geometry.rotation_matrix import rotation_matrix_to_yaw
from point_frustums.geometry.utils import angle_to_neg_pi_to_pi


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

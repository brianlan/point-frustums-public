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

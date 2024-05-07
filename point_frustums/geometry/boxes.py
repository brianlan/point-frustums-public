import torch

from point_frustums.utils.targets import Boxes
from .quaternion import apply_quaternion_to_vector, apply_quaternion_to_2d_vector, apply_quaternion_to_quaternion


def transform_boxes(boxes: Boxes, rotation: list[float], translation: list[float], rotate_first: bool = True) -> Boxes:
    """
    Apply transformation to the boxes {center, orientation, velocity}. Translate then rotate.
    :param boxes:
    :param rotation:
    :param translation:
    :param rotate_first:
    :return:
    """
    device, dtype = boxes["center"].device, boxes["center"].dtype
    rotation = torch.tensor(rotation, device=device, dtype=dtype)[None, :]
    translation = torch.tensor(translation, device=device, dtype=dtype)[None, :]

    if rotate_first:
        boxes["center"] = apply_quaternion_to_vector(q=rotation, x=boxes["center"]) + translation
    else:
        boxes["center"] = apply_quaternion_to_vector(q=rotation, x=boxes["center"] + translation)
    boxes["orientation"] = apply_quaternion_to_quaternion(rotation, boxes["orientation"])

    if boxes["velocity"].size(-1) == 3:
        boxes["velocity"] = apply_quaternion_to_vector(q=rotation, x=boxes["velocity"])
    elif boxes["velocity"].size(-1) == 2:
        boxes["velocity"] = apply_quaternion_to_2d_vector(q=rotation, x=boxes["velocity"])
    return boxes

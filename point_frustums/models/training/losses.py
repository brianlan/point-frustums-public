import torch


@torch.jit.script
def vfl(
    predictions: torch.Tensor, targets: torch.Tensor, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "none"
) -> torch.Tensor:
    """
    An adaption of the Vario-Focal Loss https://arxiv.org/abs/2008.13367

    Simplified copy of the original implementation as found here:
    https://github.com/hyz-xmaster/VarifocalNet/blob/master/mmdet/models/losses/varifocal_loss.py
    :param predictions:
    :param targets:
    :param alpha:
    :param gamma:
    :param reduction:
    :return:
    """
    # Masks for branch-less-style programming of terms 1 and 2
    t1 = torch.gt(targets, 0.0).float()
    t2 = torch.le(targets, 0.0).float()
    focal_weight = targets * t1 + alpha * (predictions.sigmoid() - targets).abs().add(1e-6).pow(gamma) * t2
    loss = focal_weight * torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "none":
        pass
    else:
        raise ValueError(f"Invalid reduction={reduction}")
    return loss

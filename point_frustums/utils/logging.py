from typing import Optional

import torch
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, Logger

from point_frustums.augmentations.augmentations_other import de_normalize
from point_frustums.config_dataclasses.dataset import Labels
from .plotting import plot_pointcloud_bev, plot_pointcloud_wandb


@torch.no_grad()
def log_pointcloud(
    logger: Logger,
    data: torch.Tensor,
    targets: Optional[dict[str, torch.Tensor]],
    detections: Optional[dict[str, torch.Tensor]],
    label_enum: Optional[Labels] = None,
    augmentations_log: Optional[dict] = None,
    tag: str = "",
    step: int = 0,
    lower_z_threshold: float = 0.0,
):
    if isinstance(augmentations_log, dict):
        pc_normalization = augmentations_log.get("Normalize", {}).get("lidar")
        if pc_normalization is not None:
            data = data.clone()
            data = de_normalize(data=data, **pc_normalization)

    if isinstance(logger, TensorBoardLogger):
        fig = plot_pointcloud_bev(points=data, targets=targets, detections=detections)
        logger.experiment.add_figure(tag, fig, step)
    elif isinstance(logger, WandbLogger):
        plot_pointcloud_wandb(
            logger,
            points=data,
            targets=targets,
            detections=detections,
            tag=tag,
            step=step,
            label_enum=label_enum,
            lower_z_threshold=lower_z_threshold,
        )

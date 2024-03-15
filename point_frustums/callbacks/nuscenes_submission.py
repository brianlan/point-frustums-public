import asyncio
import json
import os
from collections import ChainMap
from copy import deepcopy
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from point_frustums.config_dataclasses.dataset import Annotations
from point_frustums.dataloaders.nuscenes import NuScenes
from point_frustums.geometry.boxes import transform_boxes
from point_frustums.geometry.quaternion import quaternion_from_rotation_matrix


async def retrieve_boxes(
    store: dist.Store, keys: list[str], timeout: timedelta = timedelta(seconds=5)
) -> dict[str, list[dict]]:
    """
    Asynchronously retrieve all specified keys from the store.

    Samples that were not recorded will time out when trying to retrieve. An empty record will be created.
    :param store:
    :param keys:
    :param timeout:
    :return:
    """
    default_timedelta = store.timeout
    store.set_timeout(timeout)

    def get_boxes(key):
        try:
            val = json.loads(store.get(key))
        except dist.DistStoreError:
            val = []
        return {key: val}

    results = await asyncio.gather(*[asyncio.to_thread(get_boxes, k) for k in keys])

    store.set_timeout(default_timedelta)
    return dict(ChainMap(*results))


def preprocess_boxes(boxes: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Clamp the WLH box sizes to be greater zero and convert orientation to quaternion.
    :param boxes:
    :return:
    """
    boxes["wlh"].clamp_min(1e-8)
    boxes["orientation"] = quaternion_from_rotation_matrix(boxes["orientation"])
    return boxes


def postprocess_boxes(boxes: dict[str, torch.Tensor], sample_token: str, annotations: Annotations) -> list[dict]:
    """
    Convert the integer {class, attribute} labels to the string representation and re-packs the tensor format into a
    list of boxes.
    :param boxes:
    :param sample_token:
    :param annotations:
    :return:
    """
    n_boxes = boxes["score"].numel()

    score = boxes["score"].cpu().tolist()
    labels = boxes["class"].cpu().tolist()
    attributes = boxes["attribute"].cpu().tolist()
    centers = boxes["center"].cpu().tolist()
    wlhs = boxes["wlh"].cpu().tolist()
    orientations = boxes["orientation"].cpu().tolist()
    velocities = boxes["velocity"].cpu().tolist()

    boxes_list = []
    for i in range(n_boxes):
        label = annotations.classes.from_index(labels[i]).name
        attribute = annotations.attributes.from_index(attributes[i]).name
        boxes_list.append(
            {
                "sample_token": sample_token,
                "detection_score": score[i],
                "detection_name": label,
                "attribute_name": annotations.resolve_attribute(attribute=attribute, class_alias=label),
                "translation": centers[i],
                "size": wlhs[i],
                "rotation": orientations[i],
                "velocity": velocities[i],
            }
        )

    return boxes_list


def parse_sample_detections(
    sample_detections: dict[str, torch.Tensor],
    sample_metadata: dict,
    annotations: Annotations,
) -> tuple[str, list[dict]]:
    sample_detections = preprocess_boxes(deepcopy(sample_detections))
    if annotations.coos != "EGO":
        # Transform to the EGO coordinate system first if required
        sample_detections = transform_boxes(
            sample_detections,
            rotation=sample_metadata[annotations.coos]["rotation"],
            translation=sample_metadata[annotations.coos]["translation"],
        )
    # Then transform to the global coordinate system
    sample_detections = transform_boxes(
        sample_detections, rotation=sample_metadata["rotation"], translation=sample_metadata["translation"]
    )

    # Finally, create the sample record and return
    sample_token = sample_metadata["sample_token"]
    return sample_token, postprocess_boxes(sample_detections, sample_token=sample_token, annotations=annotations)


class CreateNuScenesSubmission(Callback):
    def __init__(self, use_map=False, use_external=False):
        super().__init__()
        self.use_map = use_map
        self.use_external = use_external
        self.store = None

    def on_epoch_start(self):
        if dist.is_initialized():
            master_addr = os.environ["MASTER_ADDR"]
            master_port = os.environ["MASTER_PORT"]
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            self.store = dist.TCPStore(master_addr, master_port, world_size, rank == 0)
        else:
            self.store = dist.HashStore()

    @override
    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_start()

    @override
    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_start()

    def on_batch_end(self, detections, metadata, annotations: Annotations):
        for sample_detections, sample_metadata in zip(detections, metadata):
            sample_token, sample_detections = parse_sample_detections(
                sample_detections, sample_metadata, annotations=annotations
            )
            self.store.set(sample_token, json.dumps(sample_detections))

    @override
    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs["detections"], metadata=batch["metadata"], annotations=pl_module.annotations)

    @override
    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.on_batch_end(outputs["detections"], metadata=batch["metadata"], annotations=pl_module.annotations)

    def on_epoch_end(self, trainer, dataset: NuScenes):
        if trainer.sanity_checking or bool(trainer.fast_dev_run):
            self.store = None
            return

        try:
            sample_tokens = dataset.sample_tokens
        except AttributeError as err:
            raise AttributeError("Cannot create a NuScenes submission.") from err
        all_detections = asyncio.run(retrieve_boxes(store=self.store, keys=sample_tokens))
        modalities = {s.modality for s in dataset.dataset.sensors.values()}
        submission_metadata = {
            "use_camera": "camera" in modalities,
            "use_lidar": "lidar" in modalities,
            "use_radar": "radar" in modalities,
            "use_map": self.use_map,
            "use_external": self.use_external,
        }
        submission = {"meta": submission_metadata, "results": all_detections}
        save_dir = os.path.join(trainer.logger.log_dir, "submissions")
        submission_filename = f"epoch={trainer.current_epoch}-step={trainer.global_step}.json"
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, submission_filename), "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2)

        self.store = None

    @override
    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_end(trainer, trainer.test_dataloaders.dataset)

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.on_epoch_end(trainer, trainer.val_dataloaders.dataset)

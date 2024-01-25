import json
import os
from typing import Literal

import torch
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from prettytable import PrettyTable

from point_frustums import ROOT_DIR
from point_frustums.config_dataclasses.dataset import DatasetConfig
from point_frustums.utils.environment_helpers import data_root_getter
from point_frustums.utils.geometry import (
    apply_quaternion_to_vector,
    apply_quaternion_to_2d_vector,
    apply_quaternion_to_quaternion,
    rotation_matrix_to_quaternion,
)
from .base_evaluator import Evaluator


class NuScenesEvaluator(Evaluator):
    def __init__(self, dataset: DatasetConfig, meta: dict, evaluation_dir: str):
        self.dataset = dataset
        self.annotations = dataset.annotations
        self.parsed_detections = {"meta": meta, "results": {}}
        data_root = data_root_getter(self.dataset.name, self.dataset.version, root_dir=ROOT_DIR)
        self.db = NuScenes(version=self.dataset.version, dataroot=data_root, verbose=False)
        self.evaluation_dir = evaluation_dir
        if not os.path.exists(evaluation_dir):
            os.makedirs(evaluation_dir)

    def parse(
        self,
        detections: list[
            dict[Literal["score", "class", "attribute", "center", "wlh", "orientation", "velocity"], torch.Tensor]
        ],
        metadata: dict,
    ):
        """
        Parse the detections from one batch and store them in the NuScenes EvalBoxes database.
        :param detections: The detections of each sample in the batch
        :param metadata: The metadata for the batch (mandatory, contains the transformations and sample tokens)
        :return:
        """
        for sample_detections, sample_metadata in zip(detections, metadata):
            sample_token = sample_metadata["sample_token"]
            self.parsed_detections["results"][sample_token] = []

            score = sample_detections["score"]
            class_label = sample_detections["class"]
            attribute_label = sample_detections["attribute"]
            center = sample_detections["center"]
            wlh = sample_detections["wlh"].clamp(min=1e-8)
            orientation = rotation_matrix_to_quaternion(sample_detections["orientation"])
            velocity = sample_detections["velocity"]

            # If COOS in which training targets were provided is not of the EGO (e.g. LIDAR_TOP)
            if self.annotations.coos != "EGO":
                # Transform to EGO COOS
                quat_sensor_to_ego = center.new_tensor(sample_metadata[self.annotations.coos]["rotation"])[None, :]
                translation_sensor_to_ego = center.new_tensor(sample_metadata[self.annotations.coos]["translation"])
                center = apply_quaternion_to_vector(q=quat_sensor_to_ego, x=center) + translation_sensor_to_ego[None, :]
                orientation = apply_quaternion_to_quaternion(quat_sensor_to_ego, orientation)
                velocity = apply_quaternion_to_2d_vector(q=quat_sensor_to_ego, x=velocity)

            # Save ego_translation for output
            ego_translation = center

            # Transform from the EGO to the global COOS
            quat_ego_to_global = center.new_tensor(sample_metadata["rotation"])[None, :]
            translation_ego_to_global = center.new_tensor(sample_metadata["translation"])
            center = apply_quaternion_to_vector(q=quat_ego_to_global, x=center) + translation_ego_to_global[None, :]
            orientation = apply_quaternion_to_quaternion(quat_ego_to_global, orientation)
            velocity = apply_quaternion_to_2d_vector(q=quat_ego_to_global, x=velocity)

            # Move all outputs to CPU and convert to nested lists
            score = score.cpu().tolist()
            class_label = class_label.cpu().tolist()
            attribute_label = attribute_label.cpu().tolist()
            ego_translation = ego_translation.cpu().tolist()
            center = center.cpu().tolist()
            wlh = wlh.cpu().tolist()
            orientation = orientation.cpu().tolist()
            velocity = velocity.cpu().tolist()

            for i, detection_score in enumerate(score):
                # Retrieve the names of the label indices and reconstruct the verbose form attribute
                cls = self.annotations.classes.from_index(class_label[i]).name
                attr = self.annotations.attributes.from_index(attribute_label[i]).name
                attr = self.annotations.resolve_attribute(attribute=attr, class_alias=cls)
                # Skip the intermediate step of parsing boxes and then serializing
                self.parsed_detections["results"][sample_token].append(
                    {
                        "sample_token": sample_token,
                        "ego_translation": ego_translation[i],
                        "translation": center[i],
                        "size": wlh[i],
                        "rotation": orientation[i],
                        "velocity": velocity[i],
                        "detection_name": cls,
                        "detection_score": detection_score,
                        "attribute_name": attr,
                    }
                )

    def serialize(self) -> dict:
        return self.parsed_detections

    def save(self, target_file):
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(self.parsed_detections, f, indent=2)

    def evaluate(self, epoch: int, split="val"):
        results_dir = f"{self.evaluation_dir}/{epoch=}"
        os.makedirs(results_dir, exist_ok=True)
        results_file = f"{results_dir}/results.json"
        cfg = config_factory("detection_cvpr_2019")

        if self.db.version == "v1.0-mini":
            split = "mini_" + split

        eval_engine = DetectionEval(
            nusc=self.db,
            config=cfg,
            eval_set=split,
            output_dir=os.path.expanduser(results_dir),
            result_path=os.path.expanduser(results_file),
            verbose=True,
        )

        metrics, metric_data_list = eval_engine.evaluate()
        metrics_summary = metrics.serialize()
        metrics_summary["meta"] = eval_engine.meta.copy()
        # Print high-level metrics.
        print(f"mAP: {metrics_summary['mean_ap']:.4f}")
        err_name_mapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            print(f"{err_name_mapping[tp_name]}: {tp_val:.4f}")
        print(f"NDS: {metrics_summary['nd_score']:.4f}")
        print(f"Eval time: {metrics_summary['eval_time']:.1f}s")

        # Print per-class metrics.
        print()
        print("Per-class results:")
        table = PrettyTable()
        table.field_names = ["Object Class", "AP", "ATE [m]", "ASE [1-IoU]", "AOE [rad]", "AVE [m/s]", "AAE [1-acc]"]

        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        for class_name in class_aps.keys():
            table.add_row(
                [
                    class_name,
                    class_aps[class_name],
                    class_tps[class_name]["trans_err"],
                    class_tps[class_name]["scale_err"],
                    class_tps[class_name]["orient_err"],
                    class_tps[class_name]["vel_err"],
                    class_tps[class_name]["attr_err"],
                ]
            )
        table.float_format = "3.3"
        print(table)

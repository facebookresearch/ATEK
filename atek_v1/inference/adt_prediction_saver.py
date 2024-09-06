# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from projectaria_tools.core.sophus import SE3


class AdtPredictionSaver:
    """
    Save 3D bounding box model predictions to files of the ADT format.

    Args:
        output_dir (str): directory where the predictions will be saved
        category_id_remapping_json (str): path to the json file containing the mapping from
            instance ID to category ID
    """

    def __init__(
        self,
        output_dir: str,
        metadata_file: str,
        category_id_remapping_json: str,
    ):
        self.output_dir = output_dir

        with open(metadata_file, "r") as f:
            self.instance_metadata = json.load(f)

        with open(category_id_remapping_json, "r") as f:
            inst_ids_to_cat_ids = json.load(f)
            self.cat_ids_to_inst_ids = {
                int(cat_id): int(inst_id)
                for inst_id, cat_id in inst_ids_to_cat_ids.items()
            }

    def __call__(self, prediction_list: List[List[List[Dict]]]):
        # convert prediction to pandas dataframe for processing
        flattend_prediction_list = itertools.chain(*prediction_list)
        predictions_df = pd.DataFrame(itertools.chain(*flattend_prediction_list))

        # create directory to save the predictions
        seq_name = predictions_df.iloc[0]["sequence_name"]
        pred_dir = os.path.join(self.output_dir, seq_name)
        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)

        cat_ids = predictions_df.loc[:, "category_idx"].tolist()
        object_uids = [self.cat_ids_to_inst_ids[cat_id] for cat_id in cat_ids]

        self._save_3d_bbox(predictions_df, object_uids, pred_dir)
        self._save_2d_bbox(predictions_df, object_uids, pred_dir)
        self._save_scene_objects(predictions_df, object_uids, pred_dir)
        self._save_instances(object_uids, pred_dir)

    def _save_3d_bbox(
        self, predictions_df: pd.DataFrame, object_uids: List[int], pred_dir: str
    ):
        dimensions = np.stack(predictions_df.loc[:, "dimensions"])
        timestamps = predictions_df.loc[:, "timestamp_ns"].tolist()
        dimension_data = {
            "object_uid": object_uids,
            "timestamp[ns]": timestamps,
            "p_local_obj_xmin[m]": -dimensions[:, 0] / 2,
            "p_local_obj_xmax[m]": dimensions[:, 0] / 2,
            "p_local_obj_ymin[m]": -dimensions[:, 1] / 2,
            "p_local_obj_ymax[m]": dimensions[:, 1] / 2,
            "p_local_obj_zmin[m]": -dimensions[:, 2] / 2,
            "p_local_obj_zmax[m]": dimensions[:, 2] / 2,
        }
        dimension_data = pd.DataFrame(dimension_data)
        dimension_data.to_csv(f"{pred_dir}/3d_bounding_box.csv", index=False)

    def _save_2d_bbox(
        self, predictions_df: pd.DataFrame, object_uids: List[int], pred_dir: str
    ):
        timestamps = predictions_df.loc[:, "timestamp_ns"].tolist()
        bbox_2D = np.stack(predictions_df.loc[:, "bbox_2D"])
        bbox_2d_data = {
            "stream_id": ["214-1"] * len(object_uids),
            "object_uid": object_uids,
            "timestamp[ns]": timestamps,
            "x_min[pixel]": bbox_2D[:, 0],
            "x_max[pixel]": bbox_2D[:, 2],
            "y_min[pixel]": bbox_2D[:, 1],
            "y_max[pixel]": bbox_2D[:, 3],
        }
        bbox_2d_data = pd.DataFrame(bbox_2d_data)
        bbox_2d_data.to_csv(f"{pred_dir}/2d_bounding_box.csv", index=False)

    def _save_scene_objects(
        self, predictions_df: pd.DataFrame, object_uids: List[int], pred_dir: str
    ):
        # compute object pose in world
        timestamps = predictions_df.loc[:, "timestamp_ns"].tolist()
        ts_cam_obj = np.stack(predictions_df.loc[:, "t_cam_obj"])
        Rs_cam_obj = np.stack(predictions_df.loc[:, "R_cam_obj"])
        Ts_world_cam = predictions_df.loc[:, "T_world_cam"]
        confidence = np.stack(predictions_df.loc[:, "score"])
        translations = []
        quaternions = []
        for R_cam_obj, t_cam_obj, T_world_cam in zip(
            Rs_cam_obj, ts_cam_obj, Ts_world_cam
        ):
            T_cam_obj = np.concatenate((R_cam_obj, t_cam_obj[:, np.newaxis]), axis=1)
            T_world_obj = SE3.from_matrix3x4(T_world_cam) @ SE3.from_matrix3x4(
                T_cam_obj
            )

            trans = T_world_obj.translation().squeeze()
            quat = T_world_obj.rotation().to_quat().squeeze()
            translations.append(trans)
            quaternions.append([quat[0], quat[1], quat[2], quat[3]])
        translations_np = np.array(translations)
        quaternions_np = np.array(quaternions)

        scene_objects_data = {
            "object_uid": object_uids,
            "timestamp[ns]": timestamps,
            "t_wo_x[m]": translations_np[:, 0],
            "t_wo_y[m]": translations_np[:, 1],
            "t_wo_z[m]": translations_np[:, 2],
            "q_wo_w": quaternions_np[:, 0],
            "q_wo_x": quaternions_np[:, 1],
            "q_wo_y": quaternions_np[:, 2],
            "q_wo_z": quaternions_np[:, 3],
        }
        scene_objects_data["confidence"] = confidence
        scene_objects_data = pd.DataFrame(scene_objects_data)
        scene_objects_data.to_csv(f"{pred_dir}/scene_objects.csv", index=False)

    def _save_instances(self, object_uids: List[int], pred_dir: str):
        unique_object_uids = set(object_uids)
        instances_data = {
            str(uid): self.instance_metadata[str(uid)] for uid in unique_object_uids
        }
        with open(f"{pred_dir}/instances.json", "w") as f:
            json.dump(instances_data, f, indent=2)

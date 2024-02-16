import itertools
import json
import os
from typing import Dict

import numpy as np
import pandas as pd
from projectaria_tools.core.sophus import SE3


class CubercnnPredictionSaver:
    """
    Save CubeRCNN predictions to files of the ADT format

    Args:
        config (Dict): configs needed for saving predictions, such as output_dir
    """

    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, prediction_list):
        """
        Save CubeRCNN model predictions in the same format as ADT annotations.
        """
        # load instance information and instance id to category id mapping
        with open(self.config["metadata_file"], "r") as f:
            instance_metadata = json.load(f)
        with open(self.config["category_id_remapping_json"], "r") as f:
            inst_ids_to_cat_ids = json.load(f)
        cat_ids_to_inst_ids = {
            int(cat_id): int(inst_id) for inst_id, cat_id in inst_ids_to_cat_ids.items()
        }

        # convert prediction to pandas dataframe for processing
        flattend_prediction_list = itertools.chain(*prediction_list)
        predictions_df = pd.DataFrame(itertools.chain(*flattend_prediction_list))
        ts_cam_obj = np.stack(predictions_df.loc[:, "t_cam_obj"])
        Rs_cam_obj = np.stack(predictions_df.loc[:, "R_cam_obj"])
        dimensions = np.stack(predictions_df.loc[:, "dimensions"])
        bbox_2D = np.stack(predictions_df.loc[:, "bbox_2D"])
        confidence = np.stack(predictions_df.loc[:, "score"])
        timestamps = predictions_df.loc[:, "timestamp_ns"].tolist()
        cat_ids = predictions_df.loc[:, "category_idx"].tolist()
        object_uids = [cat_ids_to_inst_ids[cat_id] for cat_id in cat_ids]

        # compute object pose in world
        Ts_world_cam = predictions_df.loc[:, "T_world_cam"]
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
        translations = np.array(translations)
        quaternions = np.array(quaternions)

        # convert data for saving
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

        scene_objects_data = {
            "object_uid": object_uids,
            "timestamp[ns]": timestamps,
            "t_wo_x[m]": translations[:, 0],
            "t_wo_y[m]": translations[:, 1],
            "t_wo_z[m]": translations[:, 2],
            "q_wo_w": quaternions[:, 0],
            "q_wo_x": quaternions[:, 1],
            "q_wo_y": quaternions[:, 2],
            "q_wo_z": quaternions[:, 3],
        }
        scene_objects_data["confidence"] = confidence

        bbox_2d_data = {
            "stream_id": ["214-1"] * len(object_uids),
            "object_uid": object_uids,
            "timestamp[ns]": timestamps,
            "x_min[pixel]": bbox_2D[:, 0],
            "x_max[pixel]": bbox_2D[:, 2],
            "y_min[pixel]": bbox_2D[:, 1],
            "y_max[pixel]": bbox_2D[:, 3],
        }

        unique_object_uids = set(object_uids)
        instances_data = {
            str(uid): instance_metadata[str(uid)] for uid in unique_object_uids
        }

        # save prediction to CSV and JSON files
        seq_name = predictions_df.iloc[0]["sequence_name"]
        pred_dir = os.path.join(self.config["output_dir"], seq_name)
        if not os.path.isdir(pred_dir):
            os.makedirs(pred_dir)
        dimension_data = pd.DataFrame(dimension_data)
        scene_objects_data = pd.DataFrame(scene_objects_data)
        bbox_2d_data = pd.DataFrame(bbox_2d_data)
        dimension_data.to_csv(f"{pred_dir}/3d_bounding_box.csv", index=False)
        scene_objects_data.to_csv(f"{pred_dir}/scene_objects.csv", index=False)
        bbox_2d_data.to_csv(f"{pred_dir}/2d_bounding_box.csv", index=False)
        with open(f"{pred_dir}/instances.json", "w") as f:
            json.dump(instances_data, f, indent=2)

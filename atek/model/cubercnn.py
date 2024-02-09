# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import itertools
import os
from argparse import Namespace
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import default_setup
from projectaria_tools.core.sophus import SE3

from atek.utils.file_utils import read_json, read_txt, write_json


def add_configs(cfg: CfgNode):
    """
    Add more options for CubeRCNN model config, based on detectron2 CfgNode
    """
    cfg.MAX_TRAINING_ATTEMPTS = 3

    cfg.TRAIN_LIST = ""
    cfg.TEST_LIST = ""
    cfg.ID_MAP_JSON = ""
    cfg.CATEGORY_JSON = ""
    cfg.SOLVER.VAL_MAX_ITER = 0


def create_config(args: Namespace) -> CfgNode:
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    # add extra configs for data
    add_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)

    return cfg


def create_cubercnn_model(args: Namespace) -> (CfgNode, torch.nn.Module):
    """
    Build CubeRCNN model architecture.
    """
    cfg = create_config(args)

    # build model and load weights
    model = build_model(cfg, priors=None)

    # load model checkpoint
    _ = DetectionCheckpointer(
        model, save_dir=os.path.dirname(args.config_file)
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    return cfg, model


def convert_cubercnn_prediction(
    data: List[Dict], prediction: List, args: Namespace, cfg: CfgNode
) -> List[Dict]:
    """
    Converts per-frame predictions to list of dicts
    """
    # assert prediction is only for one frame
    assert len(prediction) == 1

    threshold = args.threshold
    cats = cfg.DATASETS.CATEGORY_NAMES
    dets = prediction[0]["instances"]
    n_det = len(dets)
    preds_per_frame = []

    if n_det > 0:
        for (
            corners3D,
            center_cam,
            center_2D,
            dimensions,
            bbox_2D,
            pose,
            score,
            scores_full,
            cat_idx,
        ) in zip(
            dets.pred_bbox3D,
            dets.pred_center_cam,
            dets.pred_center_2D,
            dets.pred_dimensions,
            dets.pred_boxes,
            dets.pred_pose,
            dets.scores,
            dets.scores_full,
            dets.pred_classes,
        ):
            if score < threshold:
                continue
            cat = cats[cat_idx]

            predictions_dict = {
                "data_source": data[0]["data_source"],
                "sequence_name": data[0]["sequence_name"].split("/")[-3],
                "index": data[0]["index"],
                "frame_id": data[0]["frame_id"],
                "timestamp_ns": data[0]["timestamp_ns"],
                "T_world_cam": data[0]["T_world_cam"],
                "t_cam_obj": center_cam.tolist(),
                "R_cam_obj": pose.tolist(),
                # CubeRCNN dimensions are in reversed order of Aria data convention
                "dimensions": dimensions.tolist()[::-1],
                "corners3D": corners3D.tolist(),
                "center_2D": center_2D.tolist(),
                "bbox_2D": bbox_2D.tolist(),
                "score": score.detach().item(),
                "scores_full": scores_full.tolist(),
                "category_idx": cat_idx.detach().item(),
                "category": cat,
            }
            preds_per_frame.append(predictions_dict)

    return preds_per_frame


def load_bbox3d_data(bbox3d_csv):
    bbox3d_df = pd.read_csv(bbox3d_csv)
    bbox3d_data = {}
    for i in range(len(bbox3d_df)):
        row = bbox3d_df.iloc[i]
        prototype = row["prototype"]
        bbox3d_data[prototype] = np.array(
            [
                row["p_local_obj_xmin[m]"],
                row["p_local_obj_xmax[m]"],
                row["p_local_obj_ymin[m]"],
                row["p_local_obj_ymax[m]"],
                row["p_local_obj_zmin[m]"],
                row["p_local_obj_zmax[m]"],
            ]
        )
    return bbox3d_data


def move_back_object_center(bb3d_aabb, T_world_object):
    t_center_in_object = (bb3d_aabb[1::2] + bb3d_aabb[::2]) / 2
    T_object_center_bb3d = SE3.from_quat_and_translation(
        1, np.array([0, 0, 0]), -t_center_in_object
    )
    T_world_bb3d = T_world_object @ T_object_center_bb3d

    return T_world_bb3d


def save_predicted_canonical_object_poses_to_csv(
    args,
    output_csv_name,
    instance_metadata,
    object_uids,
    Rs_cam_obj,
    ts_cam_obj,
    Ts_world_cam,
    timestamps,
):
    """
    Save model-predicted object canonical 6DoF pose, which shifts model-predicted object center
    based on the prototype's canonical pose definition.
    """
    # if bbox3d_csv is available, save each object's 6DoF pose for ADT challenge submission
    if args.bbox3d_csv is None:
        return

    selected_prototypes = read_txt(args.prototype_file)
    prototypes = [instance_metadata[str(uid)]["prototype_name"] for uid in object_uids]
    bbox3d_data = load_bbox3d_data(args.bbox3d_csv)

    final_prototypes = []
    final_timestamps = []
    canonical_translations = []
    canonical_quaternions = []
    for ts, proto, R_cam_obj, t_cam_obj, T_world_cam in zip(
        timestamps, prototypes, Rs_cam_obj, ts_cam_obj, Ts_world_cam
    ):
        if proto not in selected_prototypes and proto not in bbox3d_data:
            continue
        T_cam_obj = np.concatenate((R_cam_obj, t_cam_obj[:, np.newaxis]), axis=1)
        T_cam_obj = SE3.from_matrix3x4(T_cam_obj)
        T_world_cam = SE3.from_matrix3x4(T_world_cam)
        T_world_obj = T_world_cam @ T_cam_obj

        # compute prototype canonical pose by moving object center
        # back based on canonical prototype pose definition
        T_world_obj = move_back_object_center(bbox3d_data[proto], T_world_obj)
        trans = T_world_obj.translation().squeeze()
        quat = T_world_obj.rotation().to_quat().squeeze()

        final_prototypes.append(proto)
        final_timestamps.append(ts)
        canonical_translations.append(trans)
        canonical_quaternions.append([quat[0], quat[1], quat[2], quat[3]])
    canonical_translations = np.array(canonical_translations)
    canonical_quaternions = np.array(canonical_quaternions)

    submission_data = {
        "prototype": final_prototypes,
        "timestamp_ns": final_timestamps,
        "t_wo_x": canonical_translations[:, 0],
        "t_wo_y": canonical_translations[:, 1],
        "t_wo_z": canonical_translations[:, 2],
        "q_wo_w": canonical_quaternions[:, 0],
        "q_wo_x": canonical_quaternions[:, 1],
        "q_wo_y": canonical_quaternions[:, 2],
        "q_wo_z": canonical_quaternions[:, 3],
    }
    submission_data = pd.DataFrame(submission_data)
    submission_data.to_csv(output_csv_name, index=False)


def save_cubercnn_prediction(
    dataset,
    prediction_list,
    args,
    cfg,
):
    """
    Save CubeRCNN model predictions in the same format as ADT annotations.
    """
    # load instance information and instance id to category id mapping
    instance_metadata = read_json(args.metadata_file)
    inst_ids_to_cat_ids = read_json(cfg.ID_MAP_JSON)
    cat_ids_to_inst_ids = {
        int(cat_id): int(inst_id) for inst_id, cat_id in inst_ids_to_cat_ids.items()
    }

    # convert prediction to pandas dataframe for processing
    predictions_df = pd.DataFrame(itertools.chain(*prediction_list))
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
    for R_cam_obj, t_cam_obj, T_world_cam in zip(Rs_cam_obj, ts_cam_obj, Ts_world_cam):
        T_cam_obj = np.concatenate((R_cam_obj, t_cam_obj[:, np.newaxis]), axis=1)
        T_world_obj = SE3.from_matrix3x4(T_world_cam) @ SE3.from_matrix3x4(T_cam_obj)

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
    pred_dir = os.path.join(args.output_dir, seq_name)
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    dimension_data = pd.DataFrame(dimension_data)
    scene_objects_data = pd.DataFrame(scene_objects_data)
    bbox_2d_data = pd.DataFrame(bbox_2d_data)
    dimension_data.to_csv(f"{pred_dir}/3d_bounding_box.csv", index=False)
    scene_objects_data.to_csv(f"{pred_dir}/scene_objects.csv", index=False)
    bbox_2d_data.to_csv(f"{pred_dir}/2d_bounding_box.csv", index=False)
    write_json(instances_data, f"{pred_dir}/instances.json")

    # save model prediction to ADT challenge format
    output_csv_name = f"{pred_dir}/{seq_name}.csv"
    save_predicted_canonical_object_poses_to_csv(
        args,
        output_csv_name,
        instance_metadata,
        object_uids,
        Rs_cam_obj,
        ts_cam_obj,
        Ts_world_cam,
        timestamps,
    )

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import itertools
import os

import numpy as np
import rerun as rr
import pandas as pd
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun import ToTransform3D

from atek.utils.file_utils import read_json, read_txt, write_json


TRAJECTORY_COLOR = [30, 100, 30]
GT_COLOR = [30, 200, 30]
PRED_COLOR = [200, 30, 30]


def add_configs(_C):
    _C.MAX_TRAINING_ATTEMPTS = 3

    _C.TRAIN_LIST = ""
    _C.TEST_LIST = ""
    _C.ID_MAP_JSON = ""
    _C.CATEGORY_JSON = ""
    _C.SOLVER.VAL_MAX_ITER = 0


def setup(args):
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


def build_model_with_priors(cfg, priors=None):
    model = build_model(cfg, priors=priors)
    return model


def build_cubercnn_model(args):
    cfg = setup(args)

    # build model and load weights
    model = build_model_with_priors(cfg, priors=None)
    _ = DetectionCheckpointer(
        model, save_dir=os.path.dirname(args.config_file)
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    return cfg, model


def convert_cubercnn_prediction(data, prediction, args, cfg):
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


def save_cubercnn_prediction(
    dataset,
    prediction_list,
    args,
    cfg,
):
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

    # load instance information and instance id to category id mapping
    selected_prototypes = read_txt(args.prototype_file)
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
        T_cam_obj = SE3.from_matrix3x4(T_cam_obj)
        T_world_cam = SE3.from_matrix3x4(T_world_cam)
        T_world_obj = T_world_cam @ T_cam_obj

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

    # if bbox3d_csv is available, save each object's 6DoF pose for ADT challenge submission
    if args.bbox3d_csv is not None:
        prototypes = [
            instance_metadata[str(uid)]["prototype_name"] for uid in object_uids
        ]
        bbox3d_data = load_bbox3d_data(args.bbox3d_csv)

        canonical_translations = []
        canonical_quaternions = []
        for proto, R_cam_obj, t_cam_obj, T_world_cam in zip(
            prototypes, Rs_cam_obj, ts_cam_obj, Ts_world_cam
        ):
            T_cam_obj = np.concatenate((R_cam_obj, t_cam_obj[:, np.newaxis]), axis=1)
            T_cam_obj = SE3.from_matrix3x4(T_cam_obj)
            T_world_cam = SE3.from_matrix3x4(T_world_cam)
            T_world_obj = T_world_cam @ T_cam_obj

            # compute prototype canonical pose by moving object center
            # back based on canonical prototype pose definition
            if proto in bbox3d_data:
                T_world_obj = move_back_object_center(bbox3d_data[proto], T_world_obj)
            trans = T_world_obj.translation().squeeze()
            quat = T_world_obj.rotation().to_quat().squeeze()
            canonical_translations.append(trans)
            canonical_quaternions.append([quat[0], quat[1], quat[2], quat[3]])
        canonical_translations = np.array(canonical_translations)
        canonical_quaternions = np.array(canonical_quaternions)
        submission_data = {
            "prototype": prototypes,
            "timestamp_ns": timestamps,
            "t_wo_x": canonical_translations[:, 0],
            "t_wo_y": canonical_translations[:, 1],
            "t_wo_z": canonical_translations[:, 2],
            "q_wo_w": canonical_quaternions[:, 0],
            "q_wo_x": canonical_quaternions[:, 1],
            "q_wo_y": canonical_quaternions[:, 2],
            "q_wo_z": canonical_quaternions[:, 3],
        }
        submission_data = pd.DataFrame(submission_data)
        submission_data = submission_data[
            submission_data["prototype"].isin(selected_prototypes)
        ]
        submission_data.to_csv(f"{pred_dir}/{seq_name}.csv", index=False)


class CubeRCNNViewer:
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.data_processor = dataset.data_processor
        self.camera_name = self.data_processor.vrs_camera_calib.get_label()

        rr.init("ATEK Viewer", spawn=True)
        rr.serve(web_port=args.web_port, ws_port=args.ws_port)

        # log trajectory
        trajectory = [
            self.data_processor.get_T_world_camera_by_index(i).translation()[0]
            for i in range(len(self.data_processor))
        ]
        rr.log(
            f"world/device/trajectory",
            rr.LineStrips3D(trajectory, colors=TRAJECTORY_COLOR, radii=0.01),
            timeless=True,
        )

        # log pinhole camera
        camera_calib = dataset.data_processor.final_camera_calib
        rr.log(
            f"world/device/{self.camera_name}",
            rr.Pinhole(
                resolution=[
                    int(camera_calib.get_image_size()[0]),
                    int(camera_calib.get_image_size()[1]),
                ],
                focal_length=float(camera_calib.get_focal_lengths()[0]),
            ),
            timeless=True,
        )

    def __call__(self, data, prediction, args, cfg):
        assert len(data) == 1
        index = data[0]["index"]
        ts = data[0]["timestamp_ns"]
        # print(ts)
        rr.set_time_nanos("frame_time_ns", ts)

        # process 3D and 2D bounding boxes
        T_world_cam = SE3.from_matrix3x4(data[0]["T_world_cam"])
        bb2ds_XYXY_infer = []
        labels_infer = []
        bb3ds_centers_infer = []
        bb3ds_sizes_infer = []

        if len(prediction) == 0:
            print("No prediction!")

        for pred in prediction:
            T_cam_obj_mat = np.zeros([3, 4])
            T_cam_obj_mat[0:3, 0:3] = np.array(pred["R_cam_obj"])
            T_cam_obj_mat[:, 3] = pred["t_cam_obj"]
            T_cam_obj = SE3.from_matrix3x4(T_cam_obj_mat)
            T_world_obj = T_world_cam @ T_cam_obj

            bb2ds_XYXY_infer.append(pred["bbox_2D"])
            labels_infer.append(pred["category"])
            bb3ds_centers_infer.append(T_world_obj.translation()[0])
            bb3ds_sizes_infer.append(np.array(pred["dimensions"]))

        # log camera pose
        rr.log(
            f"world/device/{self.camera_name}",
            ToTransform3D(T_world_cam, False),
        )

        # log 3D bounding boxes
        rr.log(
            f"world/device/bb3d_infer",
            rr.Boxes3D(
                sizes=bb3ds_sizes_infer,
                centers=bb3ds_centers_infer,
                radii=0.01,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

        # log image
        image = data[0]["image"].detach().cpu().numpy().transpose(1, 2, 0)
        rr.log(
            f"world/device/{self.camera_name}/image",
            rr.Image(image),
        )

        # log 2D bounding boxes
        rr.log(
            f"world/device/{self.camera_name}/bb2d_infer",
            rr.Boxes2D(
                array=bb2ds_XYXY_infer,
                array_format=rr.Box2DFormat.XYXY,
                radii=1,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

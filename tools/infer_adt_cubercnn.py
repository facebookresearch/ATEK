import argparse
import json
import os
import sys
from typing import List, Union

import cv2
import detectron2.utils.comm as comm
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import tqdm

from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor
from atek.model.cubercnn import build_model_with_priors
from cubercnn import util, vis
from cubercnn.config import get_cfg_defaults
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup, launch
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider
from torch.nn.parallel import DistributedDataParallel


def get_rgb_data_processor(data_path):
    paths_provider = AriaDigitalTwinDataPathsProvider(data_path)
    # all_device_serials = paths_provider.get_device_serial_numbers()

    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)

    rotate_image_cw90deg = True

    pose_data_processor = PoseDataProcessor(
        name="pose",
        trajectory_file=data_paths.aria_trajectory_filepath,
    )

    rgb_data_processor = FrameDataProcessor(
        video_vrs=data_paths.aria_vrs_filepath,
        stream_id=StreamId("214-1"),
        rotate_image_cw90deg=rotate_image_cw90deg,
        target_linear_camera_params=np.array([512, 512]),
        pose_data_processor=pose_data_processor,
        gt_data_processor=None,
    )

    return rgb_data_processor, data_paths


def get_K_from_frame(frame):
    cam_param = frame.camera_parameters
    K = [
        [cam_param[0], 0, cam_param[2]],
        [0, cam_param[1], cam_param[3]],
        [0, 0, 1],
    ]
    return K


def get_batch_by_index(rgb_data_processor, index: Union[int, List[int]], format="BGR"):
    if isinstance(index, int):
        index = [index]

    batched = []
    for idx in index:
        rgb_image_frame = rgb_data_processor.get_frame_by_index(idx)

        K = get_K_from_frame(rgb_image_frame)
        image = rgb_image_frame.image
        if format == "BGR":
            image = image[:, :, [2, 1, 0]]

        batched.append(
            {
                "data_source": rgb_image_frame.data_source,
                "sequence_name": rgb_image_frame.sequence_name,
                "frame_id": rgb_image_frame.frame_id,
                "timestamp_ns": rgb_image_frame.timestamp_ns,
                "T_world_cam": rgb_image_frame.T_world_camera,
                "image": torch.as_tensor(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ),
                "height": image.shape[0],
                "width": image.shape[1],
                "K": K,
            }
        )

    return batched


def add_configs(_C):
    _C.MAX_TRAINING_ATTEMPTS = 3

    _C.TRAIN_LIST = ""
    _C.TEST_LIST = ""
    _C.ID_MAP_JSON = ""
    _C.OBJ_PROP_JSON = ""
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


def export_predictions(
    predictions_df,
    pred_dir,
    cat_ids_to_inst_ids,
    object_properties,
    prototype_list,
    save_confidence=True,
):
    ts_cam_obj = np.stack(predictions_df.loc[:, "t_cam_obj"])
    Rs_cam_obj = np.stack(predictions_df.loc[:, "R_cam_obj"])
    dimensions = np.stack(predictions_df.loc[:, "dimensions"])
    bbox_2D = np.stack(predictions_df.loc[:, "bbox_2D"])
    confidence = np.stack(predictions_df.loc[:, "score"])
    scores_full = np.stack(predictions_df.loc[:, "scores_full"])

    categories = predictions_df.loc[:, "category"].tolist()
    timestamps = predictions_df.loc[:, "timestamp_ns"].tolist()

    cat_ids = predictions_df.loc[:, "category_idx"].tolist()
    object_uids = [cat_ids_to_inst_ids[cat_id] for cat_id in cat_ids]

    # convert T_cam_obj to T_world_obj, using T_world_cam from poses
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
    if save_confidence:
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
        str(uid): object_properties[str(uid)] for uid in unique_object_uids
    }

    # save data to CSV and JSON files
    dimension_data = pd.DataFrame(dimension_data)
    scene_objects_data = pd.DataFrame(scene_objects_data)
    bbox_2d_data = pd.DataFrame(bbox_2d_data)

    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    dimension_data.to_csv(f"{pred_dir}/3d_bounding_box.csv", index=False)
    scene_objects_data.to_csv(f"{pred_dir}/scene_objects.csv", index=False)
    bbox_2d_data.to_csv(f"{pred_dir}/2d_bounding_box.csv", index=False)

    with open(f"{pred_dir}/instances.json", "w") as f:
        json.dump(instances_data, f, indent=2)

    # save prediction for challenge submission
    prototypes = [object_properties[str(uid)]["prototype_name"] for uid in object_uids]
    submission_data = {
        "prototype": prototypes,
        "timestamp_ns": timestamps,
        "t_wo_x": translations[:, 0],
        "t_wo_y": translations[:, 1],
        "t_wo_z": translations[:, 2],
        "q_wo_w": quaternions[:, 0],
        "q_wo_x": quaternions[:, 1],
        "q_wo_y": quaternions[:, 2],
        "q_wo_z": quaternions[:, 3],
    }
    submission_data = pd.DataFrame(submission_data)
    submission_data = submission_data[submission_data["prototype"].isin(prototype_list)]
    seq_name = predictions_df["sequence_name"].tolist()[0]
    submission_data.to_csv(f"{pred_dir}/{seq_name}.csv", index=False)


def do_test(args, cfg, seq_data_path, model):
    # load instance information and instance id to category id mapping
    object_properties = json.load(open(cfg.OBJ_PROP_JSON, "r"))
    inst_ids_to_cat_ids = json.load(open(cfg.ID_MAP_JSON, "r"))
    cat_ids_to_inst_ids = {
        cat_id: int(inst_id) for inst_id, cat_id in inst_ids_to_cat_ids.items()
    }
    cats = cfg.DATASETS.CATEGORY_NAMES
    prototype_list = open(args.prototype_list_file, "r").read().splitlines()

    # get data
    rgb_data_processor, data_paths = get_rgb_data_processor(seq_data_path)
    seq_name = data_paths.aria_vrs_filepath.split("/")[-3]

    # create dir for predictions
    pred_dir = os.path.join(args.output_dir, seq_name)
    util.mkdir_if_missing(pred_dir)
    if args.visualize:
        vis_dir = os.path.join(pred_dir, "vis")
        util.mkdir_if_missing(vis_dir)

    # run inference
    predictions_dict_list = []
    with torch.no_grad():
        for frame_idx in range(len(rgb_data_processor)):
            batched = get_batch_by_index(rgb_data_processor, [frame_idx])
            dets = model(batched)[0]["instances"]
            n_det = len(dets)
            if n_det > 0:
                for idx, (
                    corners3D,
                    center_cam,
                    center_2D,
                    dimensions,
                    bbox_2D,
                    pose,
                    score,
                    scores_full,
                    cat_idx,
                ) in enumerate(
                    zip(
                        dets.pred_bbox3D,
                        dets.pred_center_cam,
                        dets.pred_center_2D,
                        dets.pred_dimensions,
                        dets.pred_boxes,
                        dets.pred_pose,
                        dets.scores,
                        dets.scores_full,
                        dets.pred_classes,
                    )
                ):
                    if score < args.threshold:
                        continue
                    cat = cats[cat_idx]

                    predictions_dict = {
                        "data_source": batched[0]["data_source"],
                        "sequence_name": seq_name,
                        "timestamp_ns": batched[0]["timestamp_ns"],
                        "T_world_cam": batched[0]["T_world_cam"],
                        "t_cam_obj": center_cam.tolist(),
                        "R_cam_obj": pose.tolist(),
                        # CubeRCNN dimensions are in reversed order of our conventions
                        "dimensions": dimensions.tolist()[::-1],
                        "corners3D": corners3D.tolist(),
                        "center_2D": center_2D.tolist(),
                        "bbox_2D": bbox_2D.tolist(),
                        "score": score.detach().item(),
                        "scores_full": scores_full.tolist(),
                        "category_idx": cat_idx.detach().item(),
                        "category": cat,
                    }
                    predictions_dict_list.append(predictions_dict)

            # visualize predictions
            if args.visualize and n_det > 0:
                im = batched[0]["image"].permute(1, 2, 0).numpy()[:, :, [2, 1, 0]]
                im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
                meshes = []
                meshes_text = []
                rgb_image_frame = rgb_data_processor.get_frame_by_index(frame_idx)
                K = get_K_from_frame(rgb_image_frame)
                if isinstance(K, list):
                    K = np.array(K)

                for idx, (
                    corners3D,
                    center_cam,
                    center_2D,
                    dimensions,
                    pose,
                    score,
                    cat_idx,
                ) in enumerate(
                    zip(
                        dets.pred_bbox3D,
                        dets.pred_center_cam,
                        dets.pred_center_2D,
                        dets.pred_dimensions,
                        dets.pred_pose,
                        dets.scores,
                        dets.pred_classes,
                    )
                ):
                    if score < args.threshold:
                        continue

                    cat = cats[cat_idx]
                    bbox3D = center_cam.tolist() + dimensions.tolist()
                    meshes_text.append("{} {:.2f}".format(cat, score))
                    color = [c / 255.0 for c in util.get_color(idx)]
                    box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                    meshes.append(box_mesh)

                if len(meshes) > 0:
                    im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(
                        im,
                        K,
                        meshes,
                        text=meshes_text,
                        scale=im.shape[0],
                        blend_weight=0.1,
                        blend_weight_overlay=1,
                    )
                    im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)

                # save visualized image to file
                frame_id = batched[0]["frame_id"]
                util.imwrite(
                    im_concat,
                    os.path.join(vis_dir, "{:06d}.jpg".format(frame_id)),
                )

    predictions_df = pd.DataFrame(predictions_dict_list)
    export_predictions(
        predictions_df, pred_dir, cat_ids_to_inst_ids, object_properties, prototype_list
    )
    print(f"predictions saved to {pred_dir}")


def main(args):
    print("Running on node:", os.environ["SLURM_NODEID"])
    cfg = setup(args)

    # build model and load weights
    model = build_model_with_priors(cfg, priors=None)
    _ = DetectionCheckpointer(
        model, save_dir=os.path.dirname(args.config_file)
    ).resume_or_load(cfg.MODEL.WEIGHTS, resume=True)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[comm.get_local_rank()],
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    # load test sequences
    with open(args.input_file, "r") as f:
        sequence_list = f.read().splitlines()
    world_size = comm.get_world_size()
    rank = comm.get_rank()
    sequence_list_local = sequence_list[rank::world_size]

    # test
    model = model.eval()
    for seq_data_path in sequence_list_local:
        print(f"Rank {rank}, running inference for {seq_data_path}")
        do_test(args, cfg, seq_data_path, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--input-file",
        default="",
        metavar="FILE",
        help="path to file with test sequences",
    )
    parser.add_argument(
        "--output-dir", default="", help="directory to save model predictions"
    )
    parser.add_argument("--prototype-list-file", default="", help="path to prototypes")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
        help="threshold on score for visualizing",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        action="store_true",
        help="visualize 3D boxes on image",
    )
    parser.add_argument(
        "--eval-only", default=True, action="store_true", help="perform evaluation only"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    port = (
        2**15
        + 2**14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

import os

import numpy as np

import rerun as rr
import torch

from atek.data_loaders.cubercnn_model_adaptor import (
    load_atek_wds_dataset,
    load_atek_wds_dataset_as_cubercnn,
)
from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode, Instances
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
from tqdm import tqdm

TRAJECTORY_COLOR = [30, 100, 30]
GT_COLOR = [30, 200, 30]
PRED_COLOR = [200, 30, 30]


def log_pred_3d_2d_bbox(atek_wds_dict_all):
    i_frame = 0
    for atek_wds_dict in atek_wds_dict_all:
        # Setting timestamp
        img_timestamp = atek_wds_dict["mfcd#camera-rgb+capture_timestamps_ns"][
            i_frame
        ].item()
        rr.set_time_seconds("frame_time_s", img_timestamp * 1e-9)

        T_world_device = SE3.from_matrix3x4(
            atek_wds_dict["mtd#ts_world_device"][i_frame, :, :]
        )
        T_device_cam = SE3.from_matrix3x4(
            atek_wds_dict["mfcd#camera-rgb+t_device_camera"]
        )
        # HWC -> CWH
        image = (
            atek_wds_dict["mfcd#camera-rgb+images"][i_frame]
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        # log device and camera locations
        rr.log(
            f"world",
            ToTransform3D(SE3(), False),
        )

        rr.log(
            f"world/device",
            ToTransform3D(T_world_device, False),
        )

        rr.log(
            f"world/camera-rgb",
            ToTransform3D(T_world_device @ T_device_cam, False),
        )

        # log images
        rr.log(
            f"image",
            rr.Image(image),
        )

        # For testing only

        pose_timestamp = atek_wds_dict["mtd#capture_timestamps_ns"][i_frame].item()
        # gt_timestamp = int(list(atek_wds_dict["gtdata"].keys())[i_frame])
        print(
            f"img_time: {img_timestamp}, pose_time: {pose_timestamp}, difference in us: {(img_timestamp - pose_timestamp)/1e3}"
        )

        # Log 3d bbox
        bb3ds_centers_infer = []
        bb3ds_quats_xyzw_infer = []
        bb3ds_sizes_infer = []
        labels_infer = []
        # for testing only
        print(atek_wds_dict["gtdata"].keys())
        objs = list(atek_wds_dict["gtdata"]["obb3_gt"]["bbox3d_all_instances"].values())
        for obj_gt_dict in objs:
            # Only plot chair
            if obj_gt_dict["category_id"] not in [1, 4]:
                continue
            T_world_obj = SE3.from_matrix3x4(obj_gt_dict["T_World_Object"])
            bb3ds_centers_infer.append(T_world_obj.translation()[0])
            wxyz = T_world_obj.rotation().to_quat()[0]
            bb3ds_quats_xyzw_infer.append([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])
            bb3ds_sizes_infer.append(np.array(obj_gt_dict["object_dimensions"]))
            labels_infer.append(obj_gt_dict["category_name"])

        # log 3D bounding boxes
        rr.log(
            f"world/bb3d_infer",
            rr.Boxes3D(
                sizes=bb3ds_sizes_infer,
                centers=bb3ds_centers_infer,
                rotations=bb3ds_quats_xyzw_infer,
                radii=0.01,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

        # Log 2d bbox
        bb2ds_all = []
        for obj_2d_dict in atek_wds_dict["gtdata"]["obb2_gt"]["camera-rgb"].values():
            # Only plot coffee table
            if obj_2d_dict["category_id"] not in [1, 4]:
                continue
            bb2d = obj_2d_dict["box_range"]
            bb2ds_XYXY = np.array([bb2d[0], bb2d[2], bb2d[1], bb2d[3]])
            bb2ds_all.append(bb2ds_XYXY)

        if len(bb2ds_all) == 0:
            print(f" ---- -- debug: no 2d bboxes found for frame {i_frame}")

        rr.log(
            f"image/bb2d_gt",
            rr.Boxes2D(
                array=bb2ds_all,
                array_format=rr.Box2DFormat.XYXY,
                radii=1,
                colors=GT_COLOR,
                # labels=labels_infer,
            ),
        )

    i_frame += 1


# Data visualization
rr.init("ATEK Data Loader Viewer", spawn=True)
rr.serve(web_port=8888, ws_port=8877)

# Load Native ATEK WDS data
wds_dir = "/home/louy/Calibration_data_link/Atek/2024_06_23_Test/wds_output/adt_test_1"
tars = [os.path.join(wds_dir, f"shards-000{i}.tar") for i in range(5)]

dataset = load_atek_wds_dataset(tars, batch_size=16, repeat_flag=False)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

for sample in test_dataloader:
    """
    print(f"batched sample count: {len(sample)}")
    print(sample[0].keys())
    print("Image shape: ", sample[0]['image'].shape)
    print("K: ", sample[0]['K'])
    """
    log_pred_3d_2d_bbox(sample)

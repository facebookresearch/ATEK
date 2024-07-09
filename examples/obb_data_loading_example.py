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


def log_pred_3d_2d_bbox(unbatched_atek_wds_dict):
    atek_wds_dict = unbatched_atek_wds_dict
    i_frame = 0
    # Setting timestamp
    img_timestamp = atek_wds_dict["mfcd#camera-rgb+capture_timestamps_ns"][
        i_frame
    ].item()
    rr.set_time_seconds("frame_time_s", img_timestamp * 1e-9)

    T_world_device = SE3.from_matrix3x4(
        atek_wds_dict["mtd#ts_world_device"][i_frame, :, :]
    )
    T_device_cam = SE3.from_matrix3x4(atek_wds_dict["mfcd#camera-rgb+t_device_camera"])
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

    pose_timestamp = atek_wds_dict["mtd#capture_timestamps_ns"][i_frame].item()
    print(
        f"img_time: {img_timestamp}, pose_time: {pose_timestamp}, difference in us: {(img_timestamp - pose_timestamp)/1e3}"
    )

    # Log 3d bbox
    bb3ds_centers_infer = []
    bb3ds_quats_xyzw_infer = []
    bb3ds_sizes_infer = []
    labels_infer = []

    obb3_dict = atek_wds_dict["gtdata"]["obb3_gt"]
    num_obb3 = len(obb3_dict["obb3_all_category_names"])
    for i_obj in range(num_obb3):
        # only plot chair
        if obb3_dict["obb3_all_category_names"][i_obj] not in ["chair", "sofa"]:
            continue
        T_world_obj = SE3.from_matrix3x4(
            obb3_dict["obb3_all_ts_world_object"][i_obj].numpy()
        )
        bb3ds_centers_infer.append(T_world_obj.translation()[0])
        wxyz = T_world_obj.rotation().to_quat()[0]
        bb3ds_quats_xyzw_infer.append([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])

        bb3ds_sizes_infer.append(
            np.array(obb3_dict["obb3_all_object_dimensions"][i_obj])
        )
        labels_infer.append(obb3_dict["obb3_all_category_names"][i_obj])

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

    # Log 2d bbox in camera-rgb
    bb2ds_all = []
    obb2_dict = atek_wds_dict["gtdata"]["obb2_gt"]["camera-rgb"]
    num_obb2 = len(obb2_dict["category_names"])
    for i_obj in range(num_obb2):
        # Only plot chair and sofa
        if obb2_dict["category_names"][i_obj] not in ["chair", "sofa"]:
            continue
        # re-arrange order because rerun def of bbox(XYXY) is different from ATEK(XXYY)
        bb2d = obb2_dict["box_ranges"][i_obj]
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

    # i_frame += 1


# Data visualization
rr.init("ATEK Data Loader Viewer", spawn=True)
rr.serve(web_port=8888, ws_port=8877)

# Load Native ATEK WDS data
print("-------------------- loading ATEK data natively --------------- ")
wds_dir = "/home/louy/Calibration_data_link/Atek/2024_07_02_NewGtStructure/wds_output/adt_test_1"
tars = [os.path.join(wds_dir, f"shards-000{i}.tar") for i in range(5)]

dataset = load_atek_wds_dataset(tars, batch_size=None, repeat_flag=False)
test_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

for unbatched_sample in test_dataloader:
    print(f"--- all keys in batched samples: {unbatched_sample.keys()}")
    gt_data = unbatched_sample["gtdata"]
    print(
        f"GT data type is {type(gt_data)}, len {len(gt_data)}, each sample's gt content: {gt_data.keys()}"
    )
    log_pred_3d_2d_bbox(unbatched_sample)

# Load ATEK WDS data as cubercnn
print("-------------------- loading ATEK data as CubeRCNN --------------- ")
dataset_2 = load_atek_wds_dataset_as_cubercnn(tars, batch_size=8, repeat_flag=False)
test_dataloader_2 = torch.utils.data.DataLoader(
    dataset_2,
    batch_size=None,
    num_workers=1,
    pin_memory=True,
)

for batched_sample in test_dataloader_2:

    print(f"batched_sample_size: {len(batched_sample)}")
    print(f"dict keys in a single batched sample: {batched_sample[0].keys()}")
    print(f"content in a single batched sample: {batched_sample[0]}")
    break  # only print one batched sample

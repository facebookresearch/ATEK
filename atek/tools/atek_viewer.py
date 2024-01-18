import argparse

import torch
import os
import numpy as np
import rerun as rr

from projectaria_tools.core.sophus import SE3
from projectaria_tools.core import calibration
from projectaria_tools.utils.rerun import ToTransform3D

from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataPathsProvider
)
from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor
from projectaria_tools.core.stream_id import StreamId
from atek.data_preprocess.adt_gt_data_processor import AdtGtDataProcessor
from atek.data_preprocess.frameset_aligner import FramesetAligner
from atek.data_preprocess.frameset_group_generator import FramesetGroupGenerator, FramesetSelectionConfig
from atek.dataset.atek_webdataset import create_atek_webdataset
import webdataset as wds
import yaml


# Define global variables
trajectory_color = [30, 100, 30]
eval_camera_name = "camera-rgb"
gt_color = [30, 200, 30]
infer_color = [200, 30, 30]


def logPinholeCameraData(frameset_index, camera_calib, camera_name):
    rr.log(
        f"world/device-{frameset_index}/{camera_name}",
        rr.Pinhole(
            resolution=[
                int(camera_calib.get_image_size()[0]),
                int(camera_calib.get_image_size()[1]),
            ],
            focal_length=float(
                camera_calib.get_focal_lengths()[0]
            ),
        ),
        timeless=True,
    )

def logStaticData(frameset_group_generator):
    frameset_aligner = frameset_group_generator.frameset_aligner
    num_poses = frameset_aligner.aligned_frameset_number()
    T_frameset_device = frameset_aligner.get_T_device_frameset().inverse()
    frameset_trajectory = [
        (frameset_aligner.get_T_world_frameset_by_index(index) @ T_frameset_device).translation()[0]
        for index in range(0, num_poses)
    ]
    rr.log(
        f"world/device/trajectory",
        rr.LineStrips3D(frameset_trajectory, colors=trajectory_color, radii=0.01),
        timeless=True,
    )

    # create pinhole camera models
    frameset_ids_for_groups = frameset_group_generator.frameset_ids_for_groups
    num_frames_in_group = np.shape(frameset_ids_for_groups)[1]
    for frameset_index in range(0, num_frames_in_group):
        for frame_processor in frameset_aligner.frame_data_processors:
            camera_calib = frame_processor.final_camera_calib
            camera_name = frame_processor.vrs_camera_calib.get_label()
            logPinholeCameraData(frameset_index, camera_calib, camera_name)

def logInstanceData(frameset_index, image, T_world_camera, camera_name, bb2d_XYXY, bb3d_sizes, bb3d_centers, labels, bb2d_XYXY_infer, bb3d_sizes_infer, bb3d_centers_infer, labels_infer):
   # log image
    rr.log(
        f"world/device-{frameset_index}/{camera_name}/image",
        rr.Image(image),
    )

    # log camera pose
    rr.log(
        f"world/device-{frameset_index}/{camera_name}",
        ToTransform3D(
            SE3.from_matrix3x4(T_world_camera),
            False,
        ),
    )

    # log 2d bounding box
    rr.log(
        f"world/device-{frameset_index}/{camera_name}/bb2d",
        rr.Boxes2D(
            array=bb2d_XYXY, array_format=rr.Box2DFormat.XYXY, radii=1, colors=gt_color, labels=labels
        ),
    )

   # log 3d bounding box
    rr.log(
        f"world/device-{frameset_index}/bb3d/{camera_name}",
        rr.Boxes3D(
            sizes=bb3d_sizes, centers=bb3d_centers, radii=0.01, colors=gt_color,labels = labels
        ),
    )

    # log inference data
    rr.log(
        f"world/device-{frameset_index}/{eval_camera_name}/bb2d_infer",
        rr.Boxes2D(
            array=bb2d_XYXY_infer, array_format=rr.Box2DFormat.XYXY, radii=1, colors=infer_color, labels=labels_infer
        ),
    )
    rr.log(
        f"world/device-{frameset_index}/bb3d_infer/{eval_camera_name}",
        rr.Boxes3D(
            sizes=bb3d_sizes_infer, centers=bb3d_centers_infer, radii=0.01, colors=infer_color, labels=labels_infer
        ),
    )

def logFramesetGroupData(
    frameset_group,
    eval_frame_data,
    visualize_label
):
    frameset_timestamp = frameset_group.framesets[0].timestamp_ns
    rr.set_time_nanos("frameset_timestamp", frameset_timestamp)

    for index, frameset in enumerate(frameset_group.framesets):
        for frame in frameset.frames:
            image = frame.image
            camera_name = frame.camera_name

            # gt 2d bounding box
            bb2ds = np.array(frame.bb2ds)
            class_ids = frame.object_category_ids
            labels = [frame.category_id_to_name[ids] for ids in class_ids]
            bb2d_XYXY = []
            if visualize_label is False:
                labels = []
            if frame.bb2ds is not []:
                if bb2ds.ndim == 1:
                    bb2ds = np.array([bb2ds])
                if bb2ds.shape[1] != 4:
                    print(f"2d bounding box wrong size {bb2ds}")
                    continue
                bb2d_XYXY = np.array([bb2ds[:, 0], bb2ds[:, 2], bb2ds[:, 1], bb2ds[:, 3]]).transpose()

            # gt 3d bounding box
            bb3d_sizes = []
            bb3d_centers = []
            if frame.object_dimensions is not []:
                bb3d_sizes = np.array([dim for dim in frame.object_dimensions])
                bb3d_centers = np.array([pose[:, 3] for pose in frame.Ts_world_object])

            # inference data
            bb2ds_XYXY_infer = []
            labels_infer = []
            bb3ds_centers_infer = []
            bb3ds_sizes_infer = []
            frame_id = frameset.frames[0].frame_id
            if eval_frame_data is not [] and frame_id in eval_frame_data.keys():

                class_ids = []
                for eval_frame in eval_frame_data[frame_id]:
                    bb2ds_XYXY_infer.append(eval_frame['bbox_2D'])
                    labels_infer.append(eval_frame['category'])

                    T_world_camera = SE3.from_matrix3x4(eval_frame["T_world_cam"])
                    T_camera_obj_mat = np.zeros([3, 4])
                    T_camera_obj_mat[0:3, 0:3] = np.array([list for list in eval_frame["R_cam_obj"]]).reshape(3, 3)
                    T_camera_obj_mat[:, 3] = eval_frame["t_cam_obj"]
                    T_camera_obj = SE3.from_matrix3x4(T_camera_obj_mat)
                    T_world_obj = T_world_camera @ T_camera_obj

                    bb3ds_centers_infer.append(T_world_obj.translation()[0])
                    bb3ds_sizes_infer.append(np.array(eval_frame["dimensions"]))

                if visualize_label is False:
                    labels_infer = []

            logInstanceData(index, image, frame.T_world_camera, camera_name, bb2d_XYXY, bb3d_sizes, bb3d_centers, labels, bb2ds_XYXY_infer, bb3ds_sizes_infer, bb3ds_centers_infer, labels_infer)



def setup_atek(adt_path: str):
    paths_provider = AriaDigitalTwinDataPathsProvider(adt_path)

    selected_device_number = 0
    data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)

    rgb_adt_gt_data_processor = AdtGtDataProcessor("adt_gt", "ADT", StreamId("214-1"), data_paths)
    slaml_adt_gt_data_processor = AdtGtDataProcessor("adt_gt", "ADT", StreamId("1201-1"), data_paths)
    slamr_adt_gt_data_processor = AdtGtDataProcessor("adt_gt", "ADT", StreamId("1201-2"), data_paths)

    pose_data_processor = PoseDataProcessor(
        name = "pose",
        trajectory_file = data_paths.aria_trajectory_filepath,
    )

    rotate_image_cw90deg=True

    rgb_data_processor = FrameDataProcessor(
        video_vrs=data_paths.aria_vrs_filepath,
        stream_id=StreamId("214-1"),
        rotate_image_cw90deg = rotate_image_cw90deg,
        target_linear_camera_params = np.array([512,512]),
        pose_data_processor = pose_data_processor,
        gt_data_processor = rgb_adt_gt_data_processor,
    )

    slaml_data_processor = FrameDataProcessor(
        video_vrs=data_paths.aria_vrs_filepath,
        stream_id=StreamId("1201-1"),
        rotate_image_cw90deg = rotate_image_cw90deg,
        target_linear_camera_params = np.array([320,240]),
        pose_data_processor = pose_data_processor,
        gt_data_processor = slaml_adt_gt_data_processor,
    )

    slamr_data_processor = FrameDataProcessor(
        video_vrs=data_paths.aria_vrs_filepath,
        stream_id=StreamId("1201-2"),
        rotate_image_cw90deg = rotate_image_cw90deg,
        target_linear_camera_params = np.array([320,240]),
        # target_linear_camera_params = None,
        pose_data_processor = pose_data_processor,
        gt_data_processor = slamr_adt_gt_data_processor,
    )

    frame_data_processors = [rgb_data_processor, slaml_data_processor, slamr_data_processor]
    frameset_aligner = FramesetAligner(
        target_hz = 10,
        frame_data_processors = frame_data_processors,
        pose_data_processor = pose_data_processor,
        require_objects = True)


    frameset_selection_config = FramesetSelectionConfig()
    frameset_selection_config.translation_m_threshold = 1
    frameset_selection_config.rotation_deg_threshold = 10
    frameset_group_generator = FramesetGroupGenerator(frameset_aligner=frameset_aligner,
                                                        frameset_selection_config=frameset_selection_config,
                                                        require_objects=True)

    return frameset_group_generator

def loadEvalData(data_path):
    if os.path.exists(data_path) is False:
        print(f"Cannot find eval data {data_path}")
        return []
    eval_data = torch.load(data_path)
    eval_frame_data = {}
    frame_data = []
    prev_frame_id = -1
    for data in eval_data:
        if prev_frame_id != data["frame_id"]:
            if frame_data != []:
                eval_frame_data[data["frame_id"]] = frame_data
            frame_data = []
            prev_frame_id = data["frame_id"]
        elif prev_frame_id == data["frame_id"]:
            frame_data.append(data)
    return eval_frame_data

def run_viewer( frameset_group_generator, eval_frame_data, visualize_label: bool):
    # log static data such as trajectory and point clouds
    logStaticData(frameset_group_generator)

    # log instance data such as images per frame and gt data
    num_frameset_groups = frameset_group_generator.frameset_group_number()
    for index in range(0, num_frameset_groups):
        logFramesetGroupData(frameset_group_generator.get_frameset_group_by_index(index), eval_frame_data, visualize_label)

def load_wds_data(wds_yaml):
    tar_files = []
    with open(wds_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
        data_dir = os.path.dirname(wds_yaml)
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    dataset = create_atek_webdataset(
        urls=tar_files,
        batch_size=None,
        nodesplitter=wds.shardlists.split_by_node,
        select_key_fn=None,
        remap_key_fn=None,
        data_transform_fn=None,
    )
    return dataset

def run_wds_viewer(dataset, visualize_label):
    wds_camera_names = ['camera-rgb', 'camera-slam-left', 'camera-slam-right']
    wds_camera_streamids = ['214-1', '1201-1', '1201-2']

    # log static data
    trajectory = []
    for obj in dataset:
        T_world_local = SE3.from_matrix3x4(obj["FSG+T_world_local"])
        T_local_frameset = SE3.from_matrix3x4(obj["FSG+Ts_local_frameset"][-1])
        T_world_frameset = T_world_local @ T_local_frameset
        trajectory.append(T_world_frameset.translation()[0])
        rr.log(
            f"world/device/trajectory",
            rr.LineStrips3D(trajectory, colors=trajectory_color, radii=0.01),
            timeless=True,
        )

        for camera_name, streamid in zip(wds_camera_names, wds_camera_streamids):
            images = obj[f"f#{streamid}+image"]
            for frameset_index in range(images.shape[0]):
                camera_params = obj[f"F#{streamid}+camera_parameters"][frameset_index]
                focal = camera_params[0]
                camera_calib = calibration.get_linear_camera_calibration(images.shape[2], images.shape[3], focal, camera_name)
                logPinholeCameraData(frameset_index, camera_calib, camera_name)


    # log instance data
    for obj in dataset:
        fsg_timestamp = obj['FS+timestamp_ns']
        rr.set_time_nanos("frameset_timestamp", fsg_timestamp[0])

        for camera_name, streamid in zip(wds_camera_names, wds_camera_streamids):
            images_fsg = obj[f"f#{streamid}+image"]
            num_framesets = images_fsg.shape[0]

            for frameset_index in range(num_framesets):
                image = images_fsg[frameset_index, ::].permute([1,2,0])
                # gt 2d bounding box
                bb2ds = obj[f"F#{streamid}+bb2ds"][frameset_index]
                class_ids = obj[f"F#{streamid}+object_category_ids"][frameset_index]
                category_id_to_name = obj[f"F#{streamid}+category_id_to_name"][frameset_index]
                labels = [category_id_to_name[str(ids)] for ids in class_ids]

                bb2d_XYXY = []
                if visualize_label is False:
                    labels = []
                if bb2ds is not []:
                    if bb2ds.ndim == 1:
                        bb2ds = np.array([bb2ds])
                    if bb2ds.shape[1] != 4:
                        print(f"2d bounding box wrong size {bb2ds}")
                        continue
                    bb2d_XYXY = np.array([bb2ds[:, 0], bb2ds[:, 2], bb2ds[:, 1], bb2ds[:, 3]]).transpose()

                Ts_world_camera = obj[f"F#{streamid}+T_world_camera"]
                T_world_camera = Ts_world_camera[frameset_index]

                # gt 3d bounding box
                bb3d_sizes = []
                bb3d_centers = []
                object_dimensions = obj[f"F#{streamid}+object_dimensions"][frameset_index]
                Ts_world_object = obj[f"F#{streamid}+Ts_world_object"][frameset_index]
                if object_dimensions is not []:
                    bb3d_sizes = np.array([dim for dim in object_dimensions])
                    bb3d_centers = np.array([pose[:, 3] for pose in Ts_world_object])
                logInstanceData(frameset_index, image, T_world_camera, camera_name, bb2d_XYXY, bb3d_sizes, bb3d_centers, labels, [], [], [], [])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adt-path",
        type=str,
        required=False,
        help="path to data",
    )

    parser.add_argument(
        "--wds",
        type=str,
        required=False,
        help="yaml file that contains a list of webdataset")

    parser.add_argument(
        "--infer",
        type=str,
        help="path to inference tensor data (.pth)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="path to save the rerun_recording.rrd",
    )

    parser.add_argument("--label", action="store_true", help="Visualize bounding box labels in 3D viewer")

    args = parser.parse_args()

    spawn_bool = True
    if args.output is not None and os.path.exists(args.output):
        rr.save(os.path.join(args.output, "rerun_recording.rrd"))
        spawn_bool=False
    # view webdataset
    rr.init("ATEK Viewer", spawn=spawn_bool)

    if args.wds is not None:
        print(f"ATEK viewer set up with wds input {args.wds}")
        dataset = load_wds_data(args.wds)
        run_wds_viewer(dataset, args.label)
    elif args.adt_path is not None:
        print(f"ATEK viewer set up with adt input {args.adt_path} and inference results {args.infer}")
        frameset_group_generator = setup_atek(args.adt_path)
        eval_frame_data = loadEvalData(args.infer)

        run_viewer(frameset_group_generator, eval_frame_data,  args.label)


if __name__ == "__main__":
    main()

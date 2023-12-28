import argparse

import numpy as np
import rerun as rr

from projectaria_tools.core.sophus import SE3

from projectaria_tools.utils.rerun import ToTransform3D

from projectaria_tools.projects.adt import (
   AriaDigitalTwinDataPathsProvider
)
from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor
from projectaria_tools.core.stream_id import StreamId
from atek.data_preprocess.adt_gt_data_processor import AdtGtDataProcessor
from atek.data_preprocess.frameset_aligner import FramesetAligner

# Define global variables
trajectory_color = [30, 100, 30]

def logStaticData(frameset_aligner):
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
    for frame_processor in frameset_aligner.frame_data_processors:
        camera_calib = frame_processor.final_camera_calib
        camera_name = frame_processor.vrs_camera_calib.get_label()
        rr.log(
            f"world/device/{camera_name}",
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


def logInstanceData(
    frameset_aligner,
    index,
    visualize_label
):
    frameset = frameset_aligner.get_frameset_by_index(index)
    frameset_timestamp = frameset_aligner.get_frameset_timestamp_by_index(index)
    for frame in frameset.frames:
        image = frame.image
        camera_name = frame.camera_name
        rr.set_time_nanos("frameset_timestamp", frameset_timestamp)
        # log image
        rr.log(
            f"world/device/{camera_name}/image",
            rr.Image(image),
        )

        # log camera pose
        rr.log(
            f"world/device/{camera_name}",
            ToTransform3D(
                SE3.from_matrix3x4(frame.T_world_camera),
                False,
            ),
        )

        # log 2d bounding box
        class_ids = frame.object_category_ids
        labels = [ frame.category_id_to_name[ids] for ids in class_ids]
        if frame.bb2ds is not []:
            bb2ds = np.array(frame.bb2ds)
            if bb2ds.ndim == 1:
                bb2ds = np.array([bb2ds])
            if bb2ds.shape[1] != 4:
                print(f"2d bounding box wrong size {bb2ds}")
                continue
            bb2ds_XYXY =  np.array([bb2ds[:,0], bb2ds[:, 2], bb2ds[:, 1], bb2ds[:, 3]]).transpose()
            rr.log(
                f"world/device/{camera_name}",
                rr.Boxes2D(
                    array=bb2ds_XYXY, array_format=rr.Box2DFormat.XYXY,radii=1, labels=labels,class_ids=class_ids
                ),
            )

        # log 3d bounding box
        if frame.object_dimensions is not []:
            bb3d_sizes = np.array([dim for dim in frame.object_dimensions])
            bb3d_centers = np.array([pose[:, 3] for pose in frame.Ts_world_object])
            if visualize_label is False:
                labels = []
            rr.log(
                f"world/device/trajectory",
                rr.Boxes3D(
                    sizes=bb3d_sizes, centers=bb3d_centers, radii=0.01,labels=labels,class_ids=class_ids
                ),
            )

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

    return frameset_aligner

def run_frameset_viewer(frameset_aligner : FramesetAligner, visuzlize_label: bool):
    rr.init("ATEK Viewer", spawn=True)

    # log static data such as trajectory and point clouds
    logStaticData(frameset_aligner)

    # log instance data such as images per frame and gt data
    num_frames = frameset_aligner.aligned_frameset_number()
    for index in range(0, num_frames):
        logInstanceData(frameset_aligner, index, visuzlize_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adt-path",
        type=str,
        required=True,
        help="path to data",
    )
    parser.add_argument("--label", action="store_true", help="Visualize bounding box labels in 3D viewer")
    args = parser.parse_args()
    frameset_aligner = setup_atek(args.adt_path)
    run_frameset_viewer(frameset_aligner, args.label)

if __name__ == "__main__":
    main()

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging

import numpy as np
import rerun as rr
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
)
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NativeAtekSampleVisualizer:
    """
    A visualizer for `atek_data_sample` in dict format.
    """

    COLOR_GREEN = [30, 255, 30]
    COLOR_RED = [255, 30, 30]
    COLOR_BLUE = [30, 30, 255]
    full_traj = []
    AXIS_LENGTH = 0.5
    MAX_OBB_PER_BATCH = 30  # max number of obb2d/obb3d per entity can render with label

    # max id of obb id in each batch, we record this since the entity in
    # the previous rendering batch will not be flushed if the next batch has less entities
    PREV_MAX_BB3D_ID = 0
    PREV_MAX_BB2D_ID = 0

    # max x, y,z limit for visualizing points in semidense point cloud
    # TODO: add to config file
    MAX_LIMIT_TO_VIZ_IN_SEMIDENSE_POINTS = 10

    OBB_LABELS_TO_IGNORE = [
        "other"
    ]  # if the label is in the list, we will not render it

    def __init__(
        self, viz_prefix: str = "", viz_web_port: int = 8888, viz_ws_port: int = 8899
    ) -> None:
        """
        Args:
            viz_web_port (int): port number for the Rerun web server
            viz_ws_port (int): port number for the Rerun websocket server
            viz_prefix (str): a prefix to add to re-run visualization name
        """
        self.viz_prefix = viz_prefix
        rr.init(f"ATEK Sample Viewer - {self.viz_prefix}", spawn=True)
        rr.serve(web_port=viz_web_port, ws_port=viz_ws_port)
        return

    def plot_atek_sample(
        self,
        atek_data_sample: AtekDataSample,
        plot_line_color=COLOR_GREEN,
        suffix="",  # TODO: change to a better name for suffix
    ) -> None:
        """

        plot an atek data sample instance in ReRun, including camera data, mps trajectory
        and semidense points data, and GT data. Currently supported GT data viz includes: obb3, obb2.
        """
        if not atek_data_sample:
            logger.debug(
                "ATEK data sample is empty, please check if the data is loaded correctly"
            )
            return

        self.plot_multi_frame_camera_data(atek_data_sample.camera_rgb)
        self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_left)
        self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_right)

        self.plot_mps_traj_data(atek_data_sample.mps_traj_data)
        self.plot_semidense_point_cloud(atek_data_sample.mps_semidense_point_data)

        # GT data needs timestamps associated with them
        # TODO: maybe add timestamps to GT data? Handle this in a better way
        self.plot_gtdata(
            atek_data_sample.gt_data,
            atek_data_sample.camera_rgb.capture_timestamps_ns[0].item(),
            plot_line_color=plot_line_color,
            suffix=suffix,
        )

    def plot_gtdata(self, atek_gt_dict, timestamp_ns, plot_line_color, suffix) -> None:
        if not atek_gt_dict:
            logger.debug(
                f"ATEK GT data is empty, please check if the data is loaded correctly,\
            will skip visualize GT for timestamp {timestamp_ns}"
            )
            return
        if "obb2_gt" in atek_gt_dict:
            self.plot_obb2_gt(
                atek_gt_dict["obb2_gt"],
                timestamp_ns=timestamp_ns,
                plot_color=plot_line_color,
                suffix=suffix,
            )

        if "obb3_gt" in atek_gt_dict:
            self.plot_obb3_gt(
                atek_gt_dict["obb3_gt"],
                timestamp_ns=timestamp_ns,
                plot_color=plot_line_color,
                suffix=suffix,
            )

        if "efm_gt" in atek_gt_dict:
            self.plot_efm_gt(
                atek_gt_dict["efm_gt"],
                plot_color=plot_line_color,
                suffix=suffix,
            )

    def plot_multi_frame_camera_data(self, camera_data: MultiFrameCameraData) -> None:
        if not camera_data:
            print(
                "Multiframe camera data is empty, please check if the data is loaded correctly"
            )
            return
        # Some time-invariant variables
        camera_label = camera_data.camera_label
        T_Device_Camera = SE3.from_matrix3x4(camera_data.T_Device_Camera)

        # loop over all frames
        for i_frame in range(len(camera_data.capture_timestamps_ns)):
            # Setting timestamp
            img_timestamp = camera_data.capture_timestamps_ns[i_frame].item()
            rr.set_time_seconds("frame_time_s", img_timestamp * 1e-9)

            # Plot image
            # HWC -> CWH
            image = camera_data.images[i_frame].detach().cpu().permute(1, 2, 0).numpy()
            rr.log(
                f"{camera_label}_image",
                rr.Image(image),
            )

        # Plot camera pose, we can keep this line, but now RGB camera pose should be very close to the device pose
        # to avoid multiple poses drawn on world/device, we will not plot poses for different cameras,
        # but keep the code incase user want to plot camera pose in the future

        # rerun_T_Device_Camera = ToTransform3D(T_Device_Camera, False)
        # rerun_T_Device_Camera.axis_length = self.AXIS_LENGTH
        # rr.log(f"world/device/{camera_label}", rerun_T_Device_Camera)

    def plot_mps_traj_data(self, mps_traj_data: MpsTrajData) -> None:
        if not mps_traj_data:
            print(
                "MPS trajectory data is empty, please check if the data is loaded correctly"
            )
            return
        # loop over all frames
        for i_frame in range(len(mps_traj_data.capture_timestamps_ns)):
            # Setting timestamp
            timestamp = mps_traj_data.capture_timestamps_ns[i_frame].item()
            rr.set_time_seconds("frame_time_s", timestamp * 1e-9)
            converted_world_transform = ToTransform3D(SE3(), False)
            converted_world_transform.axis_length = self.AXIS_LENGTH
            # Plot MPS trajectory
            T_World_Device = SE3.from_matrix3x4(mps_traj_data.Ts_World_Device[i_frame])
            rr.log(
                f"world",
                converted_world_transform,
            )
            # I converted transform3d here to specify the axis length
            rerun_T_World_Device = ToTransform3D(T_World_Device, False)
            rerun_T_World_Device.axis_length = self.AXIS_LENGTH
            rr.log(
                "world/device",
                rerun_T_World_Device,
            )

            self.full_traj.append(T_World_Device.translation()[0])
            rr.log(
                "world/device_trajectory",
                rr.LineStrips3D(self.full_traj),
                timeless=False,
            )

    def plot_obb2_gt(self, gt_dict, timestamp_ns, plot_color, suffix) -> None:
        if not gt_dict:
            logger.debug(
                f"ATEK obb2 GT data is empty, please check if the data is loaded correctly,\
                will skip visualize obb2 GT for timestamp {timestamp_ns}"
            )
            return
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)
        # Loop over all cameras
        for camera_label, per_cam_dict in gt_dict.items():
            num_obb2 = len(per_cam_dict["category_ids"])
            bb2ds_all = []
            category_names = []
            for i_obj in range(num_obb2):
                # re-arrange order because rerun def of bbox(XYXY) is different from ATEK(XXYY)
                bb2d = per_cam_dict["box_ranges"][i_obj]
                category_name = per_cam_dict["category_names"][i_obj]
                if category_name in self.OBB_LABELS_TO_IGNORE:
                    continue
                bb2ds_XYXY = np.array([bb2d[0], bb2d[2], bb2d[1], bb2d[3]])
                bb2ds_all.append(bb2ds_XYXY)
                category_names.append(category_name)

            # the max number one instance can be plotted is 30 for bb2d in rerun. So we need to split the whole bb2d into several parts
            # visualize the bb3d, the max number one instance can be plotted is 30 for bb3d. So we need to split the whole bb3d into several parts
            batch_id = 0
            while batch_id * self.MAX_OBB_PER_BATCH < len(bb2ds_all):
                start_obb_idx = batch_id * self.MAX_OBB_PER_BATCH
                rr.log(
                    f"{camera_label}_image/bb2d_split_{batch_id}_{suffix}",
                    rr.Boxes2D(
                        array=bb2ds_all[
                            start_obb_idx : min(
                                len(bb2ds_all), start_obb_idx + self.MAX_OBB_PER_BATCH
                            )
                        ],
                        array_format=rr.Box2DFormat.XYXY,
                        radii=0.5,
                        labels=category_names[
                            start_obb_idx : min(
                                len(bb2ds_all), start_obb_idx + self.MAX_OBB_PER_BATCH
                            )
                        ],
                        colors=plot_color,
                    ),
                )
                batch_id += 1
            cur_batch_max_id = batch_id - 1
            # flash the bb2d that is plotted in the previous batches, but has larger id
            while batch_id <= self.PREV_MAX_BB2D_ID:
                rr.log(
                    f"{camera_label}_image/bb2d_split_{batch_id}_{suffix}",
                    rr.Boxes2D(
                        array=[],
                        array_format=rr.Box2DFormat.XYXY,
                        radii=0.5,
                        labels=[],
                    ),
                )
                batch_id += 1
            self.PREV_MAX_BB2D_ID = cur_batch_max_id

    def plot_obb3_gt(self, gt_dict, timestamp_ns, plot_color, suffix) -> None:
        if not gt_dict:
            logger.debug(
                f"ATEK obb3 GT data is empty, please check if the data is loaded correctly,\
                will skip visualize GT for timestamp {timestamp_ns}"
            )
            return
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)

        # These lists are the formats required by ReRun
        bb3d_sizes = []
        bb3d_centers = []
        bb3d_quats_xyzw = []
        bb3d_labels = []

        # Loop over all cameras
        for camera_label, per_cam_dict in gt_dict.items():
            num_obb3 = len(per_cam_dict["category_ids"])
            for i_obj in range(num_obb3):
                if per_cam_dict["category_names"][i_obj] in self.OBB_LABELS_TO_IGNORE:
                    continue
                # Assign obb3 pose info
                T_world_obj = SE3.from_matrix3x4(
                    per_cam_dict["ts_world_object"][i_obj].numpy()
                )
                bb3d_centers.append(T_world_obj.translation()[0])
                wxyz = T_world_obj.rotation().to_quat()[0]
                bb3d_quats_xyzw.append([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])
                bb3d_labels.append(per_cam_dict["category_names"][i_obj])
                # Assign obb3 size info
                bb3d_sizes.append(np.array(per_cam_dict["object_dimensions"][i_obj]))
            # end for i_obj

        # visualize the bb3d, the max number one instance can be plotted is 30 for bb3d. So we need to split the whole bb3d into several parts
        batch_id = 0
        while batch_id * self.MAX_OBB_PER_BATCH < len(bb3d_sizes):
            start_obb_idx = batch_id * self.MAX_OBB_PER_BATCH
            rr.log(
                f"world/bb3d_split_{batch_id}_{suffix}",
                rr.Boxes3D(
                    sizes=bb3d_sizes[
                        start_obb_idx : min(
                            len(bb3d_sizes), start_obb_idx + self.MAX_OBB_PER_BATCH
                        )
                    ],
                    centers=bb3d_centers[
                        start_obb_idx : min(
                            len(bb3d_sizes), start_obb_idx + self.MAX_OBB_PER_BATCH
                        )
                    ],
                    rotations=bb3d_quats_xyzw[
                        start_obb_idx : min(
                            len(bb3d_sizes), start_obb_idx + self.MAX_OBB_PER_BATCH
                        )
                    ],
                    labels=bb3d_labels[
                        start_obb_idx : min(
                            len(bb3d_sizes), start_obb_idx + self.MAX_OBB_PER_BATCH
                        )
                    ],
                    colors=plot_color,
                    radii=0.01,
                ),
            )
            batch_id += 1
        cur_batch_max_id = batch_id - 1
        while batch_id <= self.PREV_MAX_BB3D_ID:
            rr.log(
                f"world/bb3d_split_{batch_id}_{suffix}",
                rr.Boxes3D(
                    sizes=[],
                    centers=[],
                    rotations=[],
                    radii=0.01,
                    labels=[],
                ),
            )
            batch_id += 1
        self.PREV_MAX_BB3D_ID = cur_batch_max_id

    def plot_semidense_point_cloud(self, mps_semidense_point_data) -> None:
        if not mps_semidense_point_data:
            logger.debug(
                "ATEK semidense point cloud data is empty, please check if the data is loaded correctly"
            )
            return

        # loop over all frames
        for i_frame in range(len(mps_semidense_point_data.capture_timestamps_ns)):
            # Setting timestamp
            pc_timestamp_ns = mps_semidense_point_data.capture_timestamps_ns[
                i_frame
            ].item()
            rr.set_time_seconds("frame_time_s", pc_timestamp_ns * 1e-9)

            points = mps_semidense_point_data.points_world[i_frame].tolist()
            filtered_points = []
            for p in points:
                if (
                    abs(p[0]) > self.MAX_LIMIT_TO_VIZ_IN_SEMIDENSE_POINTS
                    or abs(p[1]) > self.MAX_LIMIT_TO_VIZ_IN_SEMIDENSE_POINTS
                    or abs(p[2]) > self.MAX_LIMIT_TO_VIZ_IN_SEMIDENSE_POINTS
                ):
                    continue
                filtered_points.append(p)

            rr.log(
                "world/point_cloud",
                rr.Points3D(
                    filtered_points,
                    radii=0.006,
                ),
                timeless=False,
            )
        pass

    def plot_efm_gt(self, gt_dict, plot_color, suffix) -> None:
        # EFM gt is a nested dict with "timestamp(as str) -> obb3_dict"
        for timestamp_str, obb3_dict in gt_dict.items():
            self.plot_obb3_gt(obb3_dict, int(timestamp_str), plot_color, suffix)

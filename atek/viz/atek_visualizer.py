# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
from typing import Optional

import numpy as np
import rerun as rr
import torch
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
)
from atek.util.tensor_utils import compute_bbox_corners_in_world
from omegaconf.omegaconf import DictConfig

from projectaria_tools.core.calibration import CameraModelType, CameraProjection
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NativeAtekSampleVisualizer:
    """
    A visualizer for `atek_data_sample` in dict format.
    """

    COLOR_GREEN = [30, 255, 30]
    COLOR_RED = [255, 30, 30]
    COLOR_BLUE = [30, 30, 255]
    AXIS_LENGTH = 0.5
    MAX_OBB_PER_BATCH = 30  # max number of obb2d/obb3d per entity can render with label

    # max id of obb id in each batch, we record this since the entity in
    # the previous rendering batch will not be flushed if the next batch has less entities
    PREV_MAX_BB3D_ID = 0
    PREV_MAX_BB2D_ID = 0

    # max x, y,z limit for visualizing points in semidense point cloud
    # TODO: add to config file
    MAX_LIMIT_TO_VIZ_IN_SEMIDENSE_POINTS = 10

    def __init__(
        self,
        viz_prefix: str = "",
        viz_web_port: int = 8888,
        viz_ws_port: int = 8899,
        conf: Optional[DictConfig] = None,
        plot_types=[],
        obb_labels_to_ignore=[],
        obb_labels_to_include=[],
    ) -> None:
        """
        Args:
            viz_web_port (int): port number for the Rerun web server
            viz_ws_port (int): port number for the Rerun websocket server
            viz_prefix (str): a prefix to add to re-run visualization name
        """
        self.viz_prefix = viz_prefix
        self.cameras_to_plot = []

        # Full trajectory of the device
        self.full_traj = []

        # if user want to specify plot types, they can pass a config file, other wise, we will use the default plot types
        self.plot_types = [
            "camera_rgb",
            "camera_slam_left",
            "camera_slam_right",
            "mps_traj",
            "semidense_points",
            "obb2_gt",
            "obb3_gt",
            "obb3_in_camera_view",
        ]
        if conf and conf.plot_types:
            self.plot_types = conf.plot_types
        # if the label is in the list, we will not render it
        if conf and conf.obb_labels_to_ignore:
            self.obb_labels_to_ignore = conf.obb_labels_to_ignore
        else:
            self.obb_labels_to_ignore = obb_labels_to_ignore

        if conf and conf.obb_labels_to_include:
            self.obb_labels_to_include = conf.obb_labels_to_include
        else:
            self.obb_labels_to_include = obb_labels_to_include
        rr.init(f"ATEK Sample Viewer - {self.viz_prefix}", spawn=True)
        rr.serve(web_port=viz_web_port, ws_port=viz_ws_port)
        self.obb_labels_to_include = ["chair", "table", "sofa"]
        return

    def plot_atek_sample(
        self,
        atek_data_sample: Optional[AtekDataSample],
        plot_line_color=COLOR_GREEN,
        suffix="",
    ) -> None:
        """
        Plot an ATEK data sample instance in ReRun, including camera data, mps trajectory
        and semidense points data, and GT data. User can specify which plots to generate.
        Currently supported GT data viz includes: obb3, obb2.
        """

        assert (
            atek_data_sample is not None
        ), "ATEK data sample is empty in plot_atek_sample"

        if atek_data_sample.camera_rgb and "camera_rgb" in self.plot_types:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_rgb)
        if atek_data_sample.camera_slam_left and "camera_slam_left" in self.plot_types:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_left)
        if (
            atek_data_sample.camera_slam_right
            and "camera_slam_right" in self.plot_types
        ):
            self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_right)
        if atek_data_sample.mps_traj_data and "mps_traj" in self.plot_types:
            self.plot_mps_traj_data(atek_data_sample.mps_traj_data)
        if (
            atek_data_sample.mps_semidense_point_data
            and "semidense_points" in self.plot_types
        ):
            self.plot_semidense_point_cloud(atek_data_sample.mps_semidense_point_data)
        if atek_data_sample.gt_data["obb3_gt"] and "obb3_gt" in self.plot_types:
            self.plot_obb3_gt(
                atek_data_sample.gt_data["obb3_gt"],
                timestamp_ns=atek_data_sample.camera_rgb.capture_timestamps_ns[
                    0
                ].item(),
                plot_color=plot_line_color,
                suffix=suffix,
            )
        if (
            atek_data_sample.gt_data["obb3_gt"]
            and atek_data_sample.camera_rgb
            and ("camera_rgb" in self.plot_types)
            and ("obb3_in_camera_view" in self.plot_types)
        ):
            self.plot_obb3d_in_camera_view(
                obb3d_gt_dict=atek_data_sample.gt_data["obb3_gt"]["camera-rgb"],
                camera_data=atek_data_sample.camera_rgb,
                mps_traj_data=atek_data_sample.mps_traj_data,
            )
        elif (
            atek_data_sample.gt_data["obb2_gt"]
            and atek_data_sample.gt_data["obb2_gt"]
            and ("obb2_gt" in self.plot_types)
        ):
            self.plot_obb2_gt(
                atek_data_sample.gt_data["obb2_gt"],
                timestamp_ns=atek_data_sample.camera_rgb.capture_timestamps_ns[
                    0
                ].item(),
                plot_color=plot_line_color,
                suffix=suffix,
            )
        # TODO: Yang will handle the EFM GT visualization logic in other diffs
        # if atek_data_sample.gt_data["efm_gt"] and "efm_gt" in self.plot_types:
        #     self.plot_efm_gt(
        #         atek_data_sample.gt_data["efm_gt"],
        #         plot_color=plot_line_color,
        #         suffix=suffix,
        #     )
        # self.plot_efm_gt(
        #         atek_gt_dict["efm_gt"],
        #         plot_color=plot_line_color,
        #         suffix=suffix,
        #     )

    def plot_gtdata(self, atek_gt_dict, timestamp_ns, plot_line_color, suffix) -> None:

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

    def plot_multi_frame_camera_data(
        self, camera_data: Optional[MultiFrameCameraData]
    ) -> None:
        # Some time-invariant variables
        camera_label = camera_data.camera_label
        self.cameras_to_plot.append(camera_label)

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

    def plot_mps_traj_data(self, mps_traj_data: Optional[MpsTrajData]) -> None:

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

    def plot_semidense_point_cloud(
        self, mps_semidense_point_data: Optional[MpsSemiDensePointData]
    ) -> None:

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

    def plot_obb2_gt(self, gt_dict, timestamp_ns, plot_color, suffix) -> None:

        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)
        # Loop over all cameras
        for camera_label, per_cam_dict in gt_dict.items():
            if camera_label not in self.cameras_to_plot:
                continue
            num_obb2 = len(per_cam_dict["category_ids"])
            bb2ds_all = []
            category_names = []
            for i_obj in range(num_obb2):
                # re-arrange order because rerun def of bbox(XYXY) is different from ATEK(XXYY)
                bb2d = per_cam_dict["box_ranges"][i_obj]
                category_name = per_cam_dict["category_names"][i_obj].split(":")[0]
                if category_name in self.obb_labels_to_ignore:
                    continue
                if (
                    len(self.obb_labels_to_include) > 0
                    and category_name not in self.obb_labels_to_include
                ):
                    continue
                bb2ds_XYXY = np.array([bb2d[0], bb2d[2], bb2d[1], bb2d[3]])
                bb2ds_all.append(bb2ds_XYXY)
                category_names.append(
                    per_cam_dict["category_names"][i_obj]
                )  # we append this instead of category_name to keep confidence score

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
                category_name = per_cam_dict["category_names"][i_obj].split(":")[0]
                if category_name in self.obb_labels_to_ignore:
                    continue
                if (
                    len(self.obb_labels_to_include)
                    and category_name not in self.obb_labels_to_include
                ):
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

        # visualize the bb3d, the max number one instance can be plotted is 30 for bb3d.
        # So we need to split the whole bb3d into several parts
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

    def plot_obb3d_in_camera_view(
        self,
        obb3d_gt_dict: dict,
        camera_data: Optional[MultiFrameCameraData],
        mps_traj_data: Optional[MpsTrajData],
    ) -> None:
        """
        Project and plot 3D bounding box in camera view.We only support single
        timestamp in obb3d_gt_dict and camera_data, becasue input for cubercnn is single tiemstamp
        we first get all matrix needed for projection, then we calcuate the corner's position in camera view
        corner_camera = T_Device_Camera.inverse() @ (T_World_Device.inverse() @ corner)
        then we project the corner to image view using camera projection model
        """

        # we first get all matrix needed for projection
        object_dimensions = obb3d_gt_dict["object_dimensions"]
        T_World_Object = obb3d_gt_dict["ts_world_object"]
        T_Device_Camera = SE3.from_matrix3x4(camera_data.T_Device_Camera)
        camera_label = camera_data.camera_label

        # we only support single timestamp for now
        assert (
            len(mps_traj_data.Ts_World_Device) == 1
        ), "We only support single timestamp for now"
        T_World_Device = SE3.from_matrix3x4(mps_traj_data.Ts_World_Device[0])
        assert len(object_dimensions) == len(
            T_World_Object
        ), "The length of object_dimensions and T_World_Object should be the same"
        camera_model_type_dict = {
            "CameraModelType.FISHEYE624": CameraModelType.FISHEYE624,
            "CameraModelType.KANNALA_BRANDT_K3": CameraModelType.KANNALA_BRANDT_K3,
            "CameraModelType.LINEAR": CameraModelType.LINEAR,
            "CameraModelType.SPHERICAL": CameraModelType.SPHERICAL,
        }
        camera_projection = CameraProjection(
            camera_model_type_dict[camera_data.camera_model_name],
            # we should use factory calibration instead of online calibration,
            # which is more stable, meanwhile, some datasample may not have online calibration
            camera_data.projection_params.numpy(),
        )

        corners_world = compute_bbox_corners_in_world(
            object_dimensions, T_World_Object
        )  # torch.Size([num_of_instance, 8, 3])
        projected_boxes = []
        for instance in corners_world:
            projected_points = []
            for corner in instance:
                corner = np.array(corner)  # 3x1 array (3D point)
                corner_camera = T_Device_Camera.inverse() @ (
                    T_World_Device.inverse() @ corner
                )

                projected_point = camera_projection.project(corner_camera)
                # filter out boexes outside of the image boundary
                image_width, image_height = (
                    camera_data.images.shape[2],
                    camera_data.images.shape[3],
                )
                # Our image coor are on pixel center.
                if (
                    projected_point[0] < -0.5
                    or projected_point[0] > image_width - 0.5
                    or projected_point[1] < -0.5
                    or projected_point[1] > image_height - 0.5
                ):
                    continue
                projected_points.append(projected_point)

            # filter out boxes that have points outside of the image boundary
            if len(projected_points) != 8:
                continue
            projected_boxes.append(projected_points)

        timestamp_ns = camera_data.capture_timestamps_ns.item()
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)
        idx = 0  # id for the box rerun is drawing
        rr.log(f"{camera_label}_image/project3d", rr.Clear(recursive=True))
        for projected_box in projected_boxes:
            idx += 1
            rr.log(
                f"{camera_label}_image/project3d/{idx}",
                rr.LineStrips2D(
                    self._box_points_to_lines(projected_box),
                    colors=[
                        [255, 0, 0],
                        [0, 0, 255],
                        [255, 0, 0],
                        [0, 0, 255],
                        [255, 0, 0],
                        [0, 0, 255],
                        [255, 0, 0],
                        [0, 0, 255],
                        [0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 0],
                        [0, 255, 0],
                    ],
                    radii=1,
                ),
            )

    def _box_points_to_lines(self, projected_box) -> list[list[float]]:
        """
        Convert a list of 8 points to a list of lines.
        """
        p1, p2, p3, p4, p5, p6, p7, p8 = (
            projected_box[0],
            projected_box[1],
            projected_box[2],
            projected_box[3],
            projected_box[4],
            projected_box[5],
            projected_box[6],
            projected_box[7],
        )
        return [
            [p1, p2],
            [p2, p3],
            [p3, p4],
            [p4, p1],
            [p5, p6],
            [p6, p7],
            [p7, p8],
            [p8, p5],
            [p1, p5],
            [p2, p6],
            [p3, p7],
            [p4, p8],
        ]

    def plot_efm_gt(self, gt_dict, plot_color, suffix) -> None:
        # EFM gt is a nested dict with "timestamp(as str) -> obb3_dict"
        for timestamp_str, obb3_dict in gt_dict.items():
            self.plot_obb3_gt(obb3_dict, int(timestamp_str), plot_color, suffix)

    def save_viz(self, rrd_output_path: str) -> None:
        # user can use rerun [rrd_file_path] in terminal to load the visualization
        if rrd_output_path is not None:
            logger.info(f"Saving visualization to {rrd_output_path}")
            rr.save(rrd_output_path)

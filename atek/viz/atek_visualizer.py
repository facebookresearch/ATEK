# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict

import logging
from typing import Dict, List, Optional

import numpy as np
import rerun as rr
import torch
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    create_atek_data_sample_from_flatten_dict,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
)
from atek.util.tensor_utils import compute_bbox_corners_in_world
from atek.util.viz_utils import box_points_to_lines, obtain_visible_line_segs_of_obb3
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
    COLOR_GRAY = [200, 200, 200, 100]
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
        viz_web_port: Optional[int] = None,
        conf: Optional[DictConfig] = None,
        output_viz_file: Optional[str] = None,
    ) -> None:
        """
        Args:
            viz_web_port (int): port number for the Rerun web server
            viz_ws_port (int): port number for the Rerun websocket server
            viz_prefix (str): a prefix to add to re-run visualization name
        """
        self.viz_prefix = viz_prefix
        self.cameras_to_plot = []
        self.conf = conf
        self.output_viz_file = output_viz_file

        # Obb related filtering
        self.obb_labels_to_ignore = None
        self.obb_labels_to_include = None
        if "obb_viz" in conf:
            if "obb_labels_to_ignore" in conf.obb_viz:
                self.obb_labels_to_ignore = conf.obb_viz.obb_labels_to_ignore
            if "obb_labels_to_include" in conf.obb_viz:
                self.obb_labels_to_include = conf.obb_viz.obb_labels_to_include

        # Cached full trajectory of the device
        self.mps_traj_cached_full = []

        # Initializing ReRun.
        rr.init(f"ATEK Sample Viewer - {self.viz_prefix}", spawn=True)
        if viz_web_port is not None:
            rr.serve(web_port=viz_web_port, ws_port=viz_web_port + 1)
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
        self.plot_atek_sample_no_gt(atek_data_sample=atek_data_sample)

        if atek_data_sample.camera_rgb:
            timestamp_ns = atek_data_sample.camera_rgb.capture_timestamps_ns[0].item()
            self.plot_atek_sample_gt(
                atek_data_sample=atek_data_sample,
                timestamp_ns=timestamp_ns,
                plot_color=plot_line_color,
                suffix=suffix,
            )

    def plot_atek_sample_as_dict(
        self,
        atek_data_sample_dict: Dict,
        plot_line_color=COLOR_GREEN,
        suffix="",
    ) -> None:
        # Check for dict validity
        if not atek_data_sample_dict:
            logger.warning("ATEK data sample dict is empty! Not plotting this sample.")

        # Convert from flattened dict back to ATEK data sample
        atek_data_sample = create_atek_data_sample_from_flatten_dict(
            flatten_dict=atek_data_sample_dict
        )

        self.plot_atek_sample(
            atek_data_sample=atek_data_sample,
            plot_line_color=plot_line_color,
            suffix=suffix,
        )

    def plot_atek_sample_no_gt(
        self,
        atek_data_sample: Optional[AtekDataSample],
    ) -> None:
        """
        Plot an ATEK data sample instance in ReRun, including camera data, mps trajectory
        and semidense points data, but no GT data.
        """
        assert (
            atek_data_sample is not None
        ), "ATEK data sample is empty in plot_atek_sample"

        if atek_data_sample.camera_rgb:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_rgb)
        if atek_data_sample.camera_slam_left:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_left)
        if atek_data_sample.camera_slam_right:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_slam_right)

        # MPS viz
        if atek_data_sample.mps_traj_data:
            self.plot_mps_traj_data(atek_data_sample.mps_traj_data)
        if atek_data_sample.mps_semidense_point_data:
            self.plot_semidense_point_cloud(atek_data_sample.mps_semidense_point_data)

        # Depth viz
        if atek_data_sample.camera_rgb_depth:
            self.plot_multi_frame_camera_data(atek_data_sample.camera_rgb_depth)

    def plot_atek_sample_gt(
        self,
        atek_data_sample: AtekDataSample,
        timestamp_ns: int,
        plot_color: List[int],
        suffix: str = "",
    ) -> None:
        """
        Function to plot object detection GT data stored in ATEK data sample
        """
        # Plot obb3d in 3D view
        if "obb3_gt" in atek_data_sample.gt_data:
            self.plot_obb3_gt(
                atek_data_sample.gt_data["obb3_gt"],
                timestamp_ns=timestamp_ns,
                plot_color=plot_color,
                suffix=suffix,
            )

        # Plot EFM GT
        if "efm_gt" in atek_data_sample.gt_data:
            self.plot_efm_gt(
                atek_data_sample.gt_data["efm_gt"],
                plot_color=plot_color,
                suffix=suffix,
            )

        # In camera 2D view, either plot obb2d, or plot projected obb3d
        plot_projected_obb3_flag = (
            ("obb_viz" in self.conf)
            and ("plot_obb3_in_camera_view" in self.conf.obb_viz)
            and (self.conf.obb_viz.plot_obb3_in_camera_view)
        )

        if "obb2_gt" in atek_data_sample.gt_data and not plot_projected_obb3_flag:
            self.plot_obb2_gt(
                atek_data_sample.gt_data["obb2_gt"],
                timestamp_ns=timestamp_ns,
                plot_color=plot_color,
                suffix=suffix,
            )
        elif atek_data_sample.camera_rgb and plot_projected_obb3_flag:
            self.plot_obb3d_in_camera_view(
                obb3d_gt_dict=atek_data_sample.gt_data,
                camera_data=atek_data_sample.camera_rgb,
                mps_traj_data=atek_data_sample.mps_traj_data,
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
            if "depth" in camera_label:
                rr.log(
                    f"{camera_label}_image",
                    rr.DepthImage(image),
                )
            else:
                rr.log(
                    f"{camera_label}_image",
                    rr.Image(image),
                )

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

            # Cache the full trajectory
            self.mps_traj_cached_full.append(T_World_Device.translation()[0])
            rr.log(
                "world/device_trajectory",
                rr.LineStrips3D(self.mps_traj_cached_full),
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
            # Skip if this camera observation is empty
            if not per_cam_dict:
                continue

            num_obb2 = len(per_cam_dict["category_ids"])
            bb2ds_all = []
            category_names = []
            for i_obj in range(num_obb2):
                # re-arrange order because rerun def of bbox(XYXY) is different from ATEK(XXYY)
                bb2d = per_cam_dict["box_ranges"][i_obj]
                category_name = per_cam_dict["category_names"][i_obj].split(":")[0]
                if (
                    self.obb_labels_to_ignore
                    and category_name in self.obb_labels_to_ignore
                ):
                    continue
                if (
                    self.obb_labels_to_include
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
            # Skip if this camera observation is empty
            if not per_cam_dict:
                continue
            num_obb3 = len(per_cam_dict["category_ids"])
            for i_obj in range(num_obb3):
                category_name = per_cam_dict["category_names"][i_obj].split(":")[0]
                if (
                    self.obb_labels_to_ignore
                    and category_name in self.obb_labels_to_ignore
                ):
                    continue
                if (
                    self.obb_labels_to_include
                    and category_name not in self.obb_labels_to_include
                ):
                    continue
                # Assign obb3 pose info
                T_world_obj = SE3.from_matrix3x4(
                    per_cam_dict["ts_world_object"][i_obj].numpy()
                )

                bb3d_centers.append(T_world_obj.translation()[0])
                wxyz = T_world_obj.rotation().to_quat()[0]
                bb3d_quats_xyzw.append([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
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

    def _plot_obb3d_in_camera_view_single_timestamp(
        self,
        obb3d_gt_dict: dict,
        timestamp_ns: int,
        T_World_Camera: SE3,
        camera_projection: CameraProjection,
        image_width: int,
        image_height: int,
    ) -> None:
        # we first get all matrix needed for projection
        object_dimensions = obb3d_gt_dict["object_dimensions"]
        category_names = obb3d_gt_dict["category_names"]
        num_instances = len(object_dimensions)
        Ts_World_Object = obb3d_gt_dict["ts_world_object"]

        # TODO: support more camera types
        camera_label = "camera-rgb"

        # we only support single timestamp for now

        assert len(object_dimensions) == len(
            Ts_World_Object
        ), "The length of object_dimensions and T_World_Object should be the same"

        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)
        idx = 0  # id for the box rerun is drawing
        rr.log(f"{camera_label}_image/project3d", rr.Clear(recursive=True))

        corners_in_world = compute_bbox_corners_in_world(
            object_dimensions, Ts_World_Object
        )

        for i_instance in range(num_instances):
            category_name = category_names[i_instance].split(":")[0]
            if self.obb_labels_to_ignore and category_name in self.obb_labels_to_ignore:
                continue
            if (
                self.obb_labels_to_include
                and category_name not in self.obb_labels_to_include
            ):
                continue

            visible_line_segs, line_seg_colors = obtain_visible_line_segs_of_obb3(
                corners_in_world[i_instance].numpy(),
                camera_projection=camera_projection,
                T_World_Camera=T_World_Camera,
                image_width=image_width,
                image_height=image_height,
            )
            # plot edges separately
            rr.log(
                f"{camera_label}_image/project3d/instance{i_instance}",
                rr.LineStrips2D(
                    visible_line_segs,
                    colors=line_seg_colors,
                    radii=1,
                ),
            )

    def plot_obb3d_in_camera_view(
        self,
        obb3d_gt_dict: dict,
        camera_data: Optional[MultiFrameCameraData],
        mps_traj_data: Optional[MpsTrajData],
    ) -> None:
        """
        Project and plot 3D bounding box in camera view.
        we first get all matrix needed for projection, then we calcuate the corner's position in camera view
        corner_camera = T_Device_Camera.inverse() @ (T_World_Device.inverse() @ corner)
        then we project the corner to image view using camera projection model
        """
        # Get camera information
        T_Device_Camera = SE3.from_matrix3x4(camera_data.T_Device_Camera)
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
        all_timestamp_ns = camera_data.capture_timestamps_ns
        image_width = camera_data.images.shape[2]
        image_height = camera_data.images.shape[3]

        # Loop over all timestamps
        assert len(mps_traj_data.Ts_World_Device) == len(
            all_timestamp_ns
        ), "timestamp count in camera data and traj does not match."

        for i_time in range(len(all_timestamp_ns)):
            T_World_Device = SE3.from_matrix3x4(mps_traj_data.Ts_World_Device[i_time])
            timestamp_ns = all_timestamp_ns[i_time].item()
            # dict is from obb_sample_builder, single timestamp
            if "obb3_gt" in obb3d_gt_dict:
                single_obb3d_gt_dict = obb3d_gt_dict["obb3_gt"]["camera-rgb"]

            # dict is from efm_sample_builder, multi timestamp
            elif "efm_gt" in obb3d_gt_dict:
                if str(timestamp_ns) not in obb3d_gt_dict["efm_gt"]:
                    continue
                single_obb3d_gt_dict = obb3d_gt_dict["efm_gt"][str(timestamp_ns)][
                    "camera-rgb"
                ]
            else:
                logger.error(
                    f"Unsupported GT dict type: {obb3d_gt_dict.keys()}, skipping"
                )
                return

            self._plot_obb3d_in_camera_view_single_timestamp(
                obb3d_gt_dict=single_obb3d_gt_dict,
                timestamp_ns=timestamp_ns,
                T_World_Camera=T_World_Device @ T_Device_Camera,
                camera_projection=camera_projection,
                image_width=image_width,
                image_height=image_height,
            )

    def plot_efm_gt(self, gt_dict, plot_color, suffix) -> None:
        # EFM gt is a nested dict with "timestamp(as str) -> obb3_dict"
        for timestamp_str, obb3_dict in gt_dict.items():
            self.plot_obb3_gt(obb3_dict, int(timestamp_str), plot_color, suffix)

    def save_viz(self) -> None:
        # user can use rerun [rrd_file_path] in terminal to load the visualization
        if self.output_viz_file is not None:
            logger.info(f"Saving visualization to {self.output_viz_file}")
            rr.save(self.output_viz_file)

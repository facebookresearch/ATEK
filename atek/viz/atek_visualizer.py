# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import numpy as np
import rerun as rr
from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsTrajData,
    MultiFrameCameraData,
)
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D


class NativeAtekSampleVisualizer:
    """
    A visualizer for `atek_data_sample` in dict format.
    """

    COLOR_GREEN = [30, 255, 30]
    COLOR_RED = [255, 30, 30]
    COLOR_BLUE = [30, 30, 255]

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

    def plot_atek_sample(
        self,
        atek_data_sample: AtekDataSample,
        plot_line_color=COLOR_GREEN,
        suffix="",  # TODO: change to a better name for suffix
    ) -> None:
        """
        TODO: add docstring
        """
        self.plot_multi_frame_camera_data(atek_data_sample.camera_rgb)
        self.plot_mps_traj_data(atek_data_sample.mps_traj_data)

        # GT data needs timestamps associated with them
        # TODO: maybe add timestamps to GT data? Handle this in a better way
        self.plot_gtdata(
            atek_data_sample.gt_data,
            atek_data_sample.camera_rgb.capture_timestamps_ns.item(),
            plot_line_color=plot_line_color,
            suffix=suffix,
        )

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

    def plot_multi_frame_camera_data(self, camera_data: MultiFrameCameraData) -> None:
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

            # Plot camera pose
            rr.log(
                f"world/device/{camera_label}", ToTransform3D(T_Device_Camera, False)
            )

    def plot_mps_traj_data(self, mps_traj_data: MpsTrajData) -> None:
        # loop over all frames
        for i_frame in range(len(mps_traj_data.capture_timestamps_ns)):
            # Setting timestamp
            timestamp = mps_traj_data.capture_timestamps_ns[i_frame].item()
            rr.set_time_seconds("frame_time_s", timestamp * 1e-9)

            # Plot MPS trajectory
            T_World_Device = SE3.from_matrix3x4(mps_traj_data.Ts_World_Device[i_frame])
            rr.log(
                f"world",
                ToTransform3D(SE3(), False),
            )

            rr.log(
                f"world/device",
                ToTransform3D(T_World_Device, False),
            )

    def plot_obb2_gt(self, gt_dict, timestamp_ns, plot_color, suffix) -> None:
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)

        bb2ds_all = []
        # Loop over all cameras
        for camera_label, per_cam_dict in gt_dict.items():
            num_obb2 = len(per_cam_dict["category_ids"])
            for i_obj in range(num_obb2):
                # re-arrange order because rerun def of bbox(XYXY) is different from ATEK(XXYY)
                bb2d = per_cam_dict["box_ranges"][i_obj]
                bb2ds_XYXY = np.array([bb2d[0], bb2d[2], bb2d[1], bb2d[3]])
                bb2ds_all.append(bb2ds_XYXY)

            rr.log(
                f"{camera_label}_image/bb2d{suffix}",
                rr.Boxes2D(
                    array=bb2ds_all,
                    array_format=rr.Box2DFormat.XYXY,
                    radii=1,
                    colors=plot_color,
                    # labels=labels_infer, TODO: add labels_infer
                ),
            )

    def plot_obb3_gt(self, gt_dict, timestamp_ns, plot_color, suffix) -> None:
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)

        # These lists are the formats required by ReRun
        bb3d_sizes = []
        bb3d_centers = []
        bb3d_quats_xyzw = []

        # Loop over all cameras
        for _, per_cam_dict in gt_dict.items():

            num_obb3 = len(per_cam_dict["category_ids"])
            for i_obj in range(num_obb3):
                # Assign obb3 pose info
                T_world_obj = SE3.from_matrix3x4(
                    per_cam_dict["ts_world_object"][i_obj].numpy()
                )
                bb3d_centers.append(T_world_obj.translation()[0])
                wxyz = T_world_obj.rotation().to_quat()[0]
                bb3d_quats_xyzw.append([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])

                # Assign obb3 size info
                bb3d_sizes.append(np.array(per_cam_dict["object_dimensions"][i_obj]))
            # end for i_obj

            # log 3D bounding boxes
            rr.log(
                f"world/bb3d{suffix}",
                rr.Boxes3D(
                    sizes=bb3d_sizes,
                    centers=bb3d_centers,
                    rotations=bb3d_quats_xyzw,
                    radii=0.01,
                    colors=plot_color,
                    # labels=labels_infer, TODO: add labels_infer
                ),
            )

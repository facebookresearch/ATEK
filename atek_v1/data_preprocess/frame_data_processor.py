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

import copy
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd

from atek_v1.data_preprocess.base_gt_data_processor import BaseGtDataProcessor
from atek_v1.data_preprocess.data_schema import Frame
from atek_v1.data_preprocess.mps_data_processor import MpsDataProcessor
from atek_v1.utils.camera_utils import get_camera_fov_spherical_cone

from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FrameDataProcessor:
    def __init__(
        self,
        video_vrs: str,
        stream_id: StreamId,
        data_source: str = "unknown",
        time_domain: TimeDomain = TimeDomain.DEVICE_TIME,
        target_linear_camera_params: Optional[np.ndarray] = None,
        rotate_image_cw90deg: bool = True,
        mps_data_processor: Optional[MpsDataProcessor] = None,
        gt_data_processor: Optional[BaseGtDataProcessor] = None,
    ):
        """
        target_framerate: Downsample the frame rate to target_framerate. None means original rate.
        timecode_mapping: DataFrame : timecode mapping from vrs timestamps to unified timestamps
        target_linear_camera_params: Optional[np.adrray] [image_height, image_width, Optional[focal_length_xy]]
            We will use the fxfy inferred from the original camera calibration if it's not provided in target params.
        """

        self.data_source = data_source
        self.time_domain = time_domain
        self.stream_id = stream_id
        self.video_vrs = video_vrs
        self.data_provider = self.setup_data_provider()
        self.mps_data_processor = mps_data_processor

        assert (
            gt_data_processor is None or self.stream_id == gt_data_processor.stream_id
        )
        self.gt_data_processor = gt_data_processor

        # Use a data frame to hold various data for fast access and alignment.
        self.frame_df = pd.DataFrame()
        self.frame_df["timestamp_ns"] = np.array(
            self.data_provider.get_timestamps_ns(self.stream_id, self.time_domain)
        )
        self.frame_df["image_vrs_index"] = range(len(self.frame_df["timestamp_ns"]))
        self.rate_hz = self.data_provider.get_nominalRateHz(self.stream_id)

        stream_sensor_calib = self.data_provider.get_sensor_calibration(self.stream_id)
        assert (
            stream_sensor_calib is not None
        ), f"Can not get the sensor calibration for {self.stream_id} in vrs {video_vrs}"
        self.vrs_camera_calib = stream_sensor_calib.camera_calibration()
        self.final_camera_calib = self.vrs_camera_calib

        self.rotate_image_cw90deg = rotate_image_cw90deg
        if target_linear_camera_params is not None:
            # Use the fxfy inferred from the original camera calibration if it's not provided in target params.
            if len(target_linear_camera_params) > 2:
                fxfy = target_linear_camera_params[2]
            else:
                resize_ratio_W = (
                    target_linear_camera_params[0]
                    / self.vrs_camera_calib.get_image_size()[0]
                )
                resize_ratio_H = (
                    target_linear_camera_params[1]
                    / self.vrs_camera_calib.get_image_size()[1]
                )
                assert math.isclose(
                    resize_ratio_W, resize_ratio_H, rel_tol=1e-4
                ), "Image is resized by different ratio along W and H, can not infer the fxfy for target linear camera"
                fxfy = self.vrs_camera_calib.get_focal_lengths()[0] * resize_ratio_W
                logger.info(f"Inferred the fxfy of target linear camera as {fxfy}.")

            self.target_camera_calib = calibration.get_linear_camera_calibration(
                target_linear_camera_params[0],
                target_linear_camera_params[1],
                fxfy,
                f"target_linear_{self.vrs_camera_calib.get_label()}",
                self.vrs_camera_calib.get_transform_device_camera(),
            )
            self.final_camera_calib = self.target_camera_calib
        else:
            self.target_camera_calib = None

        if self.rotate_image_cw90deg:
            camera_model_to_rotate = (
                self.target_camera_calib
                if self.target_camera_calib is not None
                else self.vrs_camera_calib
            )
            assert (
                camera_model_to_rotate.model_name()
                == calibration.CameraModelType.LINEAR
            ), f"Only support Linear camera model rotation today but got {camera_model_to_rotate.model_name().name}."
            self.rotated_camera_calib = calibration.rotate_camera_calib_cw90deg(
                self.target_camera_calib
            )
            self.final_camera_calib = self.rotated_camera_calib

        self.T_device_camera = self.final_camera_calib.get_transform_device_camera()

        self.update_df_with_mps_info()
        self.cleanup_df()

        if self.gt_data_processor is not None:
            self.gt_data_processor.set_undistortion_params(
                self.vrs_camera_calib,
                self.target_camera_calib,
                self.rotate_image_cw90deg,
            )

        self.camera_fov = None

    def __len__(self):
        return len(self.frame_df)

    def cleanup_df(self):
        valid_df = self.frame_df.dropna()
        invalid_count = len(self.frame_df) - len(valid_df)
        invalid_percent = (invalid_count / len(self.frame_df)) * 100
        logger.info(
            f"Dropped {invalid_percent} percent ({invalid_count}/{len(self.frame_df)}) frames without required GT information."
        )
        self.frame_df = valid_df
        self.frame_df.reset_index(drop=True, inplace=True)

    def get_T_device_camera(self) -> SE3:
        return self.T_device_camera

    def get_rate_hz(self):
        return self.rate_hz

    def get_timestamps_ns(self):
        return self.frame_df["timestamp_ns"].values

    def update_df_with_mps_info(self):
        if self.mps_data_processor is not None:
            # add T_world_device pose info
            T_world_device_dataframe = self.mps_data_processor.get_nearest_poses(
                self.get_timestamps_ns()
            )
            self.frame_df = pd.concat([self.frame_df, T_world_device_dataframe], axis=1)

    def decide_image_subsample_factor(self, target_hz: float):
        # Frame rate sanity check.
        subsample_factor = self.rate_hz / target_hz
        assert math.isclose(subsample_factor, round(subsample_factor), rel_tol=1e-4), (
            f"Can not subsample the image stream {self.stream_id} from {self.rate_hz}hz ",
            f"to {target_hz}hz with the integer subsample factor ({subsample_factor})",
        )
        return int(round(subsample_factor))

    def setup_data_provider(self):
        provider = data_provider.create_vrs_data_provider(self.video_vrs)
        assert provider is not None, f"Cannot open {self.video_vrs}"

        options = provider.get_default_deliver_queued_options()

        all_stream_ids = options.get_stream_ids()
        if self.stream_id not in all_stream_ids:
            logger.error(
                f"{self.stream_id} is not available in {self.video_vrs} which has streams: {all_stream_ids}"
            )

        options.deactivate_stream_all()
        assert options.activate_stream(
            self.stream_id
        ), f"Can not activate stream {self.stream_id} in {self.video_vrs}"

        return provider

    def update_camera_fov_mesh(
        self,
        far_clipping_distance: float = 4.0,
        circle_segments: int = 16,
        cap_segments: int = 4,
    ):
        """
        Update and cache a watertight mesh to represent the field of view of the camera.
        This mesh could be used for rendering or camera fov overlapping check.
        """
        self.camera_fov = get_camera_fov_spherical_cone(
            camera_model=(
                self.final_camera_calib
                if self.final_camera_calib is not None
                else self.vrs_camera_calib
            ),
            far_clipping_distance=far_clipping_distance,
            circle_segments=circle_segments,
            cap_segments=cap_segments,
        )

    def get_camera_fov_mesh(
        self,
    ):
        """
        Return a watertight mesh to represent the field of view of the camera.
        This mesh could be used for rendering or camera fov overlapping check.
        """
        if self.camera_fov is None:
            # Create a default mesh for the camera fov.
            self.update_camera_fov_mesh()

        return copy.deepcopy(self.camera_fov)

    def get_frame_by_index(self, index: int) -> Frame:
        frame = Frame()
        # fill in the input image data
        frame.frame_id = int(self.frame_df["image_vrs_index"][index])
        frame.sequence_name = self.video_vrs
        frame.data_source = self.data_source

        frame.stream_id = str(self.stream_id)
        frame.camera_name = self.vrs_camera_calib.get_label()
        frame.timestamp_ns = int(self.frame_df["timestamp_ns"][index])

        image = self.data_provider.get_image_data_by_index(
            self.stream_id, frame.frame_id
        )[0].to_numpy_array()

        frame.camera_model = self.final_camera_calib.model_name().name
        frame.camera_parameters = self.final_camera_calib.projection_params()

        if self.target_camera_calib is not None:
            # Note that we need to undistort the image first then apply rotation.
            image = calibration.distort_by_calibration(
                image, self.target_camera_calib, self.vrs_camera_calib
            )
            if self.rotate_image_cw90deg:
                image = np.rot90(image, k=-1, axes=(0, 1))

        frame.image = image

        # Generate trajectory information if available.
        if "tx_world_device" in self.frame_df.columns:
            T_world_device = SE3.from_quat_and_translation(
                self.frame_df.loc[index, "qw_world_device"],
                self.frame_df.loc[
                    index, ["qx_world_device", "qy_world_device", "qz_world_device"]
                ],
                self.frame_df.loc[
                    index, ["tx_world_device", "ty_world_device", "tz_world_device"]
                ],
            )
            T_world_camera = T_world_device @ self.T_device_camera
            frame.T_world_camera = T_world_camera.to_matrix3x4()

            frame.gravity_in_world = self.frame_df.loc[
                index, ["gravity_x_world", "gravity_y_world", "gravity_z_world"]
            ].values.reshape(3, 1)
            frame.gravity_in_camera = (
                T_world_camera.rotation().inverse() @ frame.gravity_in_world
            )

        if self.gt_data_processor is not None:
            self.gt_data_processor.get_object_gt_at_timestamp_ns(
                frame, frame.timestamp_ns, T_world_camera.inverse()
            )

        return frame

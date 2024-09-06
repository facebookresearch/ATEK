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
from typing import List

import numpy as np
import pandas as pd
from atek_v1.data_preprocess.data_schema import Frameset
from atek_v1.data_preprocess.data_utils import (
    check_all_same_member,
    get_rate_stats,
    unify_object_target,
)
from atek_v1.data_preprocess.frame_data_processor import FrameDataProcessor
from atek_v1.data_preprocess.mps_data_processor import MpsDataProcessor

# from atek_v1.utils import mesh_boolean_utils

from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_stream_timestamp_key(stream_name: str) -> str:
    return f"{stream_name}_timestamp_ns"


def get_stream_index_key(stream_name: str) -> str:
    return f"{stream_name}_index"


def get_frameset_timestamp(frames_timestamp_ns: List[int], timestamp_type: str):
    """
    Return the timestamp of the frameset computed from the frames timestamp
    based on the timestamp type.
    """
    if timestamp_type == "Average":
        return round(np.mean(frames_timestamp_ns))
    elif timestamp_type == "Earliest":
        return np.min(frames_timestamp_ns)
    else:
        raise ValueError(f"Invalid timestamp type {timestamp_type}")


class FramesetAligner:
    def __init__(
        self,
        target_hz: float,
        frame_data_processors: List[FrameDataProcessor],
        mps_data_processor: MpsDataProcessor = None,
        origin_selection: str = "stream_id#214-1",
        timestamp_type: str = "Average",
        require_objects: bool = False,
    ):
        self.target_hz = target_hz
        self.frame_data_processors = frame_data_processors

        self.mps_data_processor = mps_data_processor

        self.origin_selection = origin_selection
        self.timestamp_type = timestamp_type
        self.require_objects = require_objects

        self.T_device_frameset = self.get_T_device_frameset()

        # Use a data frame to hold various data for fast access and alignment.
        self.frameset_df = self.align_images()
        self.update_df_with_timestamp_info()
        self.update_df_with_mps_info()
        self.cleanup_df()

        self.frameset_fov = None

    def get_T_device_frameset(self):
        """
        Get the device to frameset transform matrix.
        """
        if self.origin_selection.startswith("stream_id"):
            origin_stream_id = StreamId(self.origin_selection.split("#")[-1])
            for frame_data_processor in self.frame_data_processors:
                if frame_data_processor.stream_id == origin_stream_id:
                    T_device_frameset = frame_data_processor.get_T_device_camera()
        elif self.origin_selection == "device":
            T_device_frameset = SE3()
        else:
            raise ValueError(
                f"Not supported origin selection {self.origin_selection}. "
                "The valid options are 'device' or 'stream_id#<stream_id>'"
            )
        return T_device_frameset

    def get_timestamps_ns(self):
        return self.frameset_df["frameset_timestamp_ns"].values

    def update_df_with_timestamp_info(self):
        """
        Update the frameset dataframe with timestamp information.
        """
        stream_timestamp_ns_keys = [
            get_stream_timestamp_key(str(frame_data_processor.stream_id))
            for frame_data_processor in self.frame_data_processors
        ]
        srteam_timestamp_ns_data = self.frameset_df[stream_timestamp_ns_keys].values
        if self.timestamp_type == "Average":
            frameset_timestamp_ns = np.round(
                np.mean(srteam_timestamp_ns_data, axis=-1)
            ).astype(int)
        elif self.timestamp_type == "Earliest":
            frameset_timestamp_ns = np.min(srteam_timestamp_ns_data, axis=-1).astype(
                int
            )
        else:
            raise ValueError(f"Invalid timestamp type {self.timestamp_type}")

        self.frameset_df["frameset_timestamp_ns"] = frameset_timestamp_ns

    def update_df_with_mps_info(self):
        if self.mps_data_processor is not None:
            # Add trajectory info to DataFrame
            T_world_device_dataframe = self.mps_data_processor.get_nearest_poses(
                self.get_timestamps_ns()
            )
            self.frameset_df = pd.concat(
                [self.frameset_df, T_world_device_dataframe], axis=1
            )

            # add semi-dense point cloud info to DataFrame
            semidense_dataframe = self.mps_data_processor.get_nearest_semidense_points(
                self.get_timestamps_ns(), tolerance_ns=150_000_000
            )
            self.frameset_df = pd.concat(
                [self.frameset_df, semidense_dataframe], axis=1
            )

    def cleanup_df(self):
        valid_df = self.frameset_df.dropna()
        invalid_count = len(self.frameset_df) - len(valid_df)
        invalid_percent = (invalid_count / len(self.frameset_df)) * 100
        logger.info(
            f"Dropped {invalid_percent} percent ({invalid_count}/{len(self.frameset_df)}) framesets without required GT information."
        )
        self.frameset_df = valid_df
        self.frameset_df.reset_index(drop=True, inplace=True)

    def align_images(self, tolerance_ns: int = 150000):
        """
        Align the images data of different streams to the first image data stream.
        Only align the nearest images within the tolerance_ns, 150 us is a default
        empirical good number for alignment. Particular consider the default offset
        between rgb image and slam images are 100us.
        """
        # Align image timestamps
        aligned_df = None
        subsample_factor_after_alignment = None
        key_for_align = "timestamp_ns_for_align"
        for frame_data_processor in self.frame_data_processors:
            tss_ns = frame_data_processor.get_timestamps_ns()
            stream_name = str(frame_data_processor.stream_id)

            subsample_factor_after_alignment = (
                min(
                    subsample_factor_after_alignment,
                    frame_data_processor.decide_image_subsample_factor(self.target_hz),
                )
                if subsample_factor_after_alignment is not None
                else frame_data_processor.decide_image_subsample_factor(self.target_hz)
            )

            df = pd.DataFrame(
                {
                    key_for_align: tss_ns,
                    get_stream_timestamp_key(stream_name): tss_ns,
                    get_stream_index_key(stream_name): range(len(tss_ns)),
                }
            )
            if aligned_df is None:
                aligned_df = df
            else:
                aligned_df = (
                    pd.merge_asof(
                        aligned_df.sort_values(key_for_align),
                        df.sort_values(key_for_align),
                        on=key_for_align,
                        tolerance=tolerance_ns,
                        direction="nearest",
                    )
                    .dropna()
                    .astype(int)
                )

        rate_stats = get_rate_stats(aligned_df[key_for_align])
        logger.info(f"Rate status after alignment: {rate_stats}")
        assert math.isclose(
            rate_stats["rate_hz"] / subsample_factor_after_alignment,
            self.target_hz,
            rel_tol=0.01 * self.target_hz,
        )  # allow 1% error

        subsampled_aligned_df = aligned_df.iloc[::subsample_factor_after_alignment]
        rate_stats = get_rate_stats(subsampled_aligned_df[key_for_align])
        logger.info(f"Rate status after subsample: {rate_stats}")

        subsampled_aligned_df.reset_index(inplace=True, drop=True)
        return subsampled_aligned_df

    def update_frameset_fov_mesh(
        self,
        far_clipping_distance: float = 4.0,
        circle_segments: int = 16,
        cap_segments: int = 4,
    ):
        """
        Update and cache a watertight mesh in frameset coordinate
        to represent the field of view of the frameset.
        This mesh could be used for rendering or fov overlapping check.
        """
        self.frameset_fov = None
        for frame_date_processor in self.frame_data_processors:
            frame_date_processor.update_camera_fov_mesh(
                far_clipping_distance, circle_segments, cap_segments
            )
            camera_fov = frame_date_processor.get_camera_fov_mesh()
            T_device_camera = frame_date_processor.get_T_device_camera()
            T_frameset_camera = self.T_device_frameset.inverse() @ T_device_camera
            camera_fov_in_frameset = camera_fov.apply_transform(
                T_frameset_camera.to_matrix()
            )
            if self.frameset_fov is None:
                self.frameset_fov = camera_fov_in_frameset
            else:
                # TODO: temporarily disabled to pass CI. The preprocessing part of atek_v1 should NOT be used!
                """
                self.frameset_fov = mesh_boolean_utils.union_meshes(
                    self.frameset_fov, camera_fov_in_frameset
                )
                """
                pass

    def get_frameset_fov_mesh(self):
        if self.frameset_fov is None:
            # Create a default mesh for the camera fov.
            self.update_frameset_fov_mesh()

        return copy.deepcopy(self.frameset_fov)

    def aligned_frameset_number(self):
        return len(self.frameset_df)

    def check_index_valid(self, index: int):
        assert (
            0 <= index < self.aligned_frameset_number()
        ), f"Index{index} out of bound [0, {self.aligned_frameset_number})."

    def get_frameset_timestamp_by_index(self, index: int) -> int:
        """
        Function to query the frameset timestamp by index.
        """
        self.check_index_valid(index)

        return int(self.frameset_df.loc[index, "frameset_timestamp_ns"])

    def get_T_world_frameset_by_index(self, index: int) -> SE3:
        """
        Function to query the T_world_frameset by index.
        """
        self.check_index_valid(index)
        assert self.mps_data_processor is not None

        T_world_device = SE3.from_quat_and_translation(
            self.frameset_df.loc[index, "qw_world_device"],
            self.frameset_df.loc[
                index, ["qx_world_device", "qy_world_device", "qz_world_device"]
            ],
            self.frameset_df.loc[
                index, ["tx_world_device", "ty_world_device", "tz_world_device"]
            ],
        )
        T_world_frameset = T_world_device @ self.T_device_frameset

        return T_world_frameset

    def get_Ts_frameset_camera_by_index(self, index: int) -> List[SE3]:
        """
        Function to query the Ts_frameset_camera by index.
        """
        self.check_index_valid(index)

        T_frameset_device = self.T_device_frameset.inverse()
        Ts_frameset_camera = [
            (
                T_frameset_device @ frame_data_processor.get_T_device_camera()
            ).to_matrix3x4()
            for frame_data_processor in self.frame_data_processors
        ]

        return Ts_frameset_camera

    def get_frameset_by_index(self, index: int) -> Frameset:
        """
        Get the data by index from all the data processors.
        """
        self.check_index_valid(index)

        frameset = Frameset()
        frameset.frames = []
        for frame_data_processor in self.frame_data_processors:
            stream_name = str(frame_data_processor.stream_id)
            target_frame_index = int(
                self.frameset_df.loc[index, get_stream_index_key(stream_name)]
            )
            frame = frame_data_processor.get_frame_by_index(target_frame_index)
            frameset.frames.append(frame)

        # Fill in the frameset and some sanity checks
        assert len(frameset.frames) > 0, "No valid frames found for frameset."
        assert check_all_same_member(
            frameset.frames, "data_source"
        ), "All frames must be from the same data sources."
        frameset.data_source = frameset.frames[0].data_source

        assert check_all_same_member(
            frameset.frames, "sequence_name"
        ), "All frames must be from the same sequence."
        frameset.sequence_name = frameset.frames[0].sequence_name

        frameset.timestamp_type = self.timestamp_type
        frameset.timestamp_ns = self.get_frameset_timestamp_by_index(index)

        frameset.origin_selection = self.origin_selection
        frameset.Ts_frameset_camera = self.get_Ts_frameset_camera_by_index(index)

        # Generate trajectory information if available.
        if "tx_world_device" in self.frameset_df.columns:
            T_world_frameset = self.get_T_world_frameset_by_index(index)
            frameset.T_world_frameset = T_world_frameset.to_matrix3x4()
            frameset.gravity_in_world = self.frameset_df.loc[
                index, ["gravity_x_world", "gravity_y_world", "gravity_z_world"]
            ].values.reshape((3, 1))
            frameset.gravity_in_frameset = (
                T_world_frameset.rotation().inverse() @ frameset.gravity_in_world
            )

            if self.require_objects:
                unify_object_target(frameset)
                T_frameset_world = T_world_frameset.inverse()
                frameset.Ts_frameset_object = [
                    (
                        T_frameset_world @ SE3.from_matrix3x4(T_world_object)
                    ).to_matrix3x4()
                    for T_world_object in frameset.Ts_world_object
                ]

        # Generate semidense point information if available.
        if "points_world" in self.frameset_df.columns:
            frameset.points_world = self.frameset_df.iloc[index]["points_world"]
            frameset.points_dist_std = self.frameset_df.iloc[index]["points_dist_std"]

        return frameset

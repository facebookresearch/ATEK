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

import logging
from typing import Dict, List, Optional

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsTrajData,
    MultiFrameCameraData,
)
from atek.data_preprocess.processors.aria_camera_processor import AriaCameraProcessor
from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from atek.data_preprocess.processors.obb2_gt_processor import Obb2GtProcessor
from atek.data_preprocess.processors.obb3_gt_processor import Obb3GtProcessor
from omegaconf.omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ObbSampleBuilder:
    """
    A general sample builder for general 2D/3D object detection tasks.

    Models using this sample builder:
    - CubeRCNN
    """

    def __init__(
        self,
        conf: DictConfig,
        vrs_file: str,
        sequence_name: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
    ) -> None:
        """
        Initializes the ObbSampleBuilder with necessary configuration and file paths.

        Args:
            conf (DictConfig): Configuration object containing settings for various processors.
            vrs_file (str): Path to the main Aria VRS file used for video and sensor data.
            mps_files (Dict[str, str]): Dictionary mapping keys to file paths for MPS data, including:
                "mps_closedloop_traj_file": Path to the closed-loop trajectory file.
            gt_files (Dict[str, str]): Dictionary mapping keys to file paths for ground truth data, including:
                "obb3_file": Path to the 3D object bounding box file.
                "obb3_traj_file": Path to the 3D object bounding box trajectory file.
                "obb2_file": Path to the 2D object bounding box file.
                "instance_json_file": Path to the JSON file containing object instance information.
        """
        self.conf = conf

        self.vrs_file = vrs_file
        self.sequence_name = sequence_name

        self.processors = self._add_processors_from_conf(
            conf, vrs_file, mps_files if mps_files is not None else {}, gt_files
        )

    def _add_processors_from_conf(
        self,
        conf: DictConfig,
        vrs_file: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
    ):
        """
        This function creates a dict of processors from the config file.
        """
        processors = {}
        # camera processors
        camera_conf_list = [
            conf.rgb,
            conf.slam_left,
            conf.slam_right,
        ]
        # These fields will be used in creating obb2 processor
        selected_camera_label_to_stream_ids = {}
        selected_camera_label_to_pixel_transforms = {}
        selected_camera_label_to_calib = {}
        for camera_conf in camera_conf_list:
            if camera_conf.selected:
                cam_processor = AriaCameraProcessor(vrs_file, camera_conf)
                processors[camera_conf.sensor_label] = cam_processor
                selected_camera_label_to_stream_ids[camera_conf.sensor_label] = (
                    cam_processor.get_stream_id()
                )
                selected_camera_label_to_pixel_transforms[camera_conf.sensor_label] = (
                    cam_processor.get_pixel_transform()
                )
                selected_camera_label_to_calib[camera_conf.sensor_label] = (
                    cam_processor.get_final_camera_calib()
                )

        if "mps_traj" in conf and conf.mps_traj.selected:
            processors["mps_traj"] = MpsTrajProcessor(
                mps_closedloop_traj_file=mps_files["mps_closedloop_traj_file"],
                conf=conf.mps_traj,
            )

        if len(gt_files) > 0 and "obb_gt" in conf and conf.obb_gt.selected:
            # GT data contains both obb3 and obb2 data, therefore we create 2 processors.

            # Create obb3 processor
            processors["obb3_gt"] = Obb3GtProcessor(
                obb3_file_path=gt_files["obb3_file"],
                obb3_traj_file_path=gt_files["obb3_traj_file"],
                instance_json_file_path=gt_files["instance_json_file"],
                obb2_file_path=gt_files["obb2_file"],
                category_mapping_file_path=gt_files.get(
                    "category_mapping_file", None
                ),  # this file is optional
                camera_label_to_stream_ids=selected_camera_label_to_stream_ids,
                conf=conf.obb_gt,
            )

            # Create obb2 processor, which requires streamid, pixel transforms, and calib from camera processors.
            processors["obb2_gt"] = Obb2GtProcessor(
                obb2_file_path=gt_files["obb2_file"],
                instance_json_file_path=gt_files["instance_json_file"],
                category_mapping_file_path=gt_files.get(
                    "category_mapping_file", None
                ),  # this file is optional
                camera_label_to_stream_ids=selected_camera_label_to_stream_ids,
                camera_label_to_pixel_transforms=selected_camera_label_to_pixel_transforms,
                camera_label_to_calib=selected_camera_label_to_calib,
                conf=conf.obb_gt,
            )

        return processors

    def get_sample_by_timestamp_ns(self, timestamp_ns: int) -> Optional[AtekDataSample]:
        sample = AtekDataSample()

        # First assign sequence name
        sample.sequence_name = self.sequence_name

        for processor_label, processor in self.processors.items():
            if isinstance(processor, AriaCameraProcessor):
                # ========================================
                # Aria camera sensor data
                # ========================================
                sample_camera_data = processor.get_image_data_by_timestamps_ns(
                    timestamps_ns=[timestamp_ns]
                )
                # Skip if no image data is available
                if sample_camera_data is None:
                    logger.warning(
                        f"Querying camera for {timestamp_ns} on processor {processor_label} has returned None, skipping this sample."
                    )
                    return None

                # Fill calibration data
                sample_camera_data.camera_label = processor_label
                sample_camera_data.origin_camera_label = processor.get_origin_label()
                final_camera_calib = processor.get_final_camera_calib()
                sample_camera_data.T_Device_Camera = torch.from_numpy(
                    final_camera_calib.get_transform_device_camera().to_matrix3x4()
                )
                sample_camera_data.camera_model_name = str(
                    final_camera_calib.model_name()
                )
                sample_camera_data.projection_params = torch.from_numpy(
                    final_camera_calib.projection_params()
                )

                setattr(
                    sample,
                    processor_label.replace(
                        "-", "_"
                    ),  # field name in AtekDataSample is same as sensor label apart from a `-`->`_` conversion
                    sample_camera_data,
                )
            # ========================================
            # MPS traj data
            # ========================================
            elif isinstance(processor, MpsTrajProcessor):
                maybe_mps_traj_data = processor.get_closed_loop_pose_by_timestamps_ns(
                    [timestamp_ns]
                )
                if maybe_mps_traj_data is None:
                    logger.warning(
                        f"Querying MPS traj for {timestamp_ns} has returned None, skipping this sample."
                    )
                    return None

                # Fill MPS traj data into sample
                sample.mps_traj_data = MpsTrajData(
                    Ts_World_Device=maybe_mps_traj_data[0],
                    capture_timestamps_ns=maybe_mps_traj_data[1],
                    gravity_in_world=maybe_mps_traj_data[2],
                )

            # ========================================
            # GT data
            # ========================================
            elif isinstance(processor, Obb3GtProcessor):
                maybe_gt_data = processor.get_gt_by_timestamp_ns(timestamp_ns)
                if maybe_gt_data is None:
                    logger.warning(
                        f"Querying 3D Bbox GT data for {timestamp_ns} has returned None, skipping this sample."
                    )
                    return None
                sample.gt_data["obb3_gt"] = maybe_gt_data

            elif isinstance(processor, Obb2GtProcessor):
                maybe_gt_data = processor.get_gt_by_timestamp_ns(timestamp_ns)
                if maybe_gt_data is None:
                    logger.warning(
                        f"Querying 2D bbox GT data for {timestamp_ns} has returned None, skipping this sample."
                    )
                    return None
                sample.gt_data["obb2_gt"] = maybe_gt_data

            else:
                raise ValueError(
                    f"Unimplemented processor class {processor.__name__} in SampleBuilder! "
                )

        return sample

    def get_sample_by_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[AtekDataSample]:
        """
        This API allows for a List[timestamp] as input, however, the list must be of length 1.
        This is for API consistency with the other sample builders.
        """
        assert (
            len(timestamps_ns) == 1
        ), "Only support single timestamp query for ObbSampleBuilder!"
        return self.get_sample_by_timestamp_ns(timestamps_ns[0])

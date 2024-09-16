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
import os
from dataclasses import fields
from typing import Dict, List, Optional

import torch

from atek.data_preprocess.atek_data_sample import AtekDataSample, MpsTrajData
from atek.data_preprocess.processors.aria_camera_processor import AriaCameraProcessor
from atek.data_preprocess.processors.depth_image_processor import DepthImageProcessor
from atek.data_preprocess.processors.efm_gt_processor import EfmGtProcessor
from atek.data_preprocess.processors.mps_semidense_processor import (
    MpsSemiDenseProcessor,
)
from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from omegaconf.omegaconf import DictConfig
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EfmSampleBuilder:
    """
    A customized sample builder for EFM, which includes both 3D obb detection task and surface reconstruction task.

    Models using this sample builder:
    - EFM
    """

    def __init__(
        self,
        conf: DictConfig,
        vrs_file: str,
        sequence_name: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
        depth_vrs_file: str,
    ) -> None:
        """
        Initializes the EfmSampleBuilder with necessary configuration and file paths.

        Args:
            conf (DictConfig): Configuration object containing settings for various processors.
            vrs_file (str): Path to the main Aria VRS file used for video and sensor data.
            mps_files (Dict[str, str]): Dictionary mapping keys to file paths for MPS data, including:
                "mps_closedloop_traj_file": Path to the closed-loop trajectory file.
                "mps_semidense_points_file": Path to the global semidense points file.
                "mps_semidense_observations_file": Path to the observations of semidense points, indicating which points are observable by which camera, at each timestamp.
            gt_files (Dict[str, str]): Dictionary mapping keys to file paths for ground truth data, including:
                "obb3_file": Path to the 3D object bounding box file.
                "obb3_traj_file": Path to the 3D object bounding box trajectory file.
                "obb2_file": Path to the 2D object bounding box file.
                "instance_json_file": Path to the JSON file containing object instance information.
            depth_vrs_file [Optional]: Path to the depth VRS file. This is required for the surface recon task.
        """
        self.conf = conf

        self.vrs_file = vrs_file
        self.depth_vrs_file = depth_vrs_file
        self.sequence_name = sequence_name

        self.processors = self._add_processors_from_conf(
            conf, vrs_file, mps_files, gt_files
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
        selected_camera_label_to_stream_ids = {}
        for camera_conf in camera_conf_list:
            if camera_conf.selected:
                cam_processor = AriaCameraProcessor(vrs_file, camera_conf)
                processors[camera_conf.sensor_label] = cam_processor
                selected_camera_label_to_stream_ids[camera_conf.sensor_label] = (
                    cam_processor.get_stream_id()
                )

        # MPS processors
        if "mps_traj" in conf and conf.mps_traj.selected:
            processors["mps_traj"] = MpsTrajProcessor(
                mps_closedloop_traj_file=mps_files["mps_closedloop_traj_file"],
                conf=conf.mps_traj,
            )

        if "mps_semidense" in conf and conf.mps_semidense.selected:
            processors["mps_semidense"] = MpsSemiDenseProcessor(
                mps_semidense_points_file=mps_files["mps_semidense_points_file"],
                mps_semidense_observations_file=mps_files[
                    "mps_semidense_observations_file"
                ],
                conf=conf.mps_semidense,
            )

        # Depth processor
        if (
            "rgb_depth" in conf
            and conf.rgb_depth.selected
            and os.path.exists(self.depth_vrs_file)
        ):
            assert (
                self.depth_vrs_file != ""
            ), "need to specify depth vrs file to use depth processor"

            # Obtain image transformations from rgb AriaCameraProcessor, where interpolation needs to be exactly set to NEAREST
            assert (
                "camera-rgb" in processors
            ), "rgb_depth depends on camera_rgb processor to obtain camera calibration"
            depth_image_transform = processors["camera-rgb"].get_image_transform(
                rescale_interpolation=InterpolationMode.NEAREST
            )
            depth_camera_calib = processors["camera-rgb"].get_final_camera_calib()

            processors["rgb_depth"] = DepthImageProcessor(
                depth_vrs=self.depth_vrs_file,
                image_transform=depth_image_transform,
                depth_camera_calib=depth_camera_calib,
                depth_camera_label="camera-rgb-depth",
                conf=conf.rgb_depth,
            )

        if len(gt_files) > 0 and "efm_gt" in conf and conf.efm_gt.selected:
            processors["efm_gt"] = EfmGtProcessor(
                obb3_file_path=gt_files["obb3_file"],
                obb3_traj_file_path=gt_files["obb3_traj_file"],
                obb2_file_path=gt_files["obb2_file"],
                instance_json_file_path=gt_files["instance_json_file"],
                category_mapping_file_path=gt_files.get(
                    "category_mapping_file", None
                ),  # this file is optional
                camera_label_to_stream_ids=selected_camera_label_to_stream_ids,
                conf=conf.efm_gt,
            )

        return processors

    def _check_timestamps_are_consistent(
        self, sample: AtekDataSample, timestamps_ns: List[int]
    ) -> bool:
        """
        Check if all timestamps in the sample are consistent.
        """
        num_timestamps = len(timestamps_ns)
        for data_field in fields(sample):
            name = data_field.name
            data_field_value = getattr(sample, name)
            if hasattr(data_field_value, "capture_timestamps_ns"):
                time = getattr(data_field_value, "capture_timestamps_ns")
                if len(time) != num_timestamps:
                    logger.warning(
                        f"Timestamps in {name} does not have full count of {num_timestamps}!"
                    )
                    return False

            # Check for gt
            if name == "gt_data" and "efm_gt" in data_field_value:
                if len(data_field_value["efm_gt"]) != num_timestamps:
                    logger.warning(
                        f"Timestamps in {name} does not have full count of {num_timestamps}!"
                    )
                    return False

        return True

    def get_sample_by_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[AtekDataSample]:
        sample = AtekDataSample()

        # First assign sequence name
        sample.sequence_name = self.sequence_name

        for processor_label, processor in self.processors.items():
            if isinstance(processor, AriaCameraProcessor):
                # ========================================
                # Aria camera sensor data
                # ========================================
                sample_camera_data = processor.get_image_data_by_timestamps_ns(
                    timestamps_ns=timestamps_ns
                )
                # Skip if no image data is available
                if sample_camera_data is None:
                    logger.warning(
                        f"Querying camera for {timestamps_ns} on processor {processor_label} has returned None, skipping this sample."
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
                sample_camera_data.camera_valid_radius = torch.tensor(
                    [final_camera_calib.get_valid_radius()], dtype=torch.float32
                )

                setattr(
                    sample,
                    processor_label.replace(
                        "-", "_"
                    ),  # field name in AtekDataSample is same as sensor label, apart from a `-`->`_` conversion
                    sample_camera_data,
                )

            # ========================================
            # MPS traj data
            # ========================================
            elif isinstance(processor, MpsTrajProcessor):
                maybe_mps_traj_data = processor.get_closed_loop_pose_by_timestamps_ns(
                    timestamps_ns
                )
                if maybe_mps_traj_data is None:
                    logger.warning(
                        f"Querying MPS traj for {timestamps_ns} has returned None, skipping this sample."
                    )
                    return None

                # Fill MPS traj data into sample
                sample.mps_traj_data = MpsTrajData(
                    Ts_World_Device=maybe_mps_traj_data[0],
                    capture_timestamps_ns=maybe_mps_traj_data[1],
                    gravity_in_world=maybe_mps_traj_data[2],
                )

            # =======================================
            # MPS SemiDense data
            # =======================================
            elif isinstance(processor, MpsSemiDenseProcessor):
                maybe_mps_semidense_data = (
                    processor.get_semidense_points_by_timestamps_ns(timestamps_ns)
                )
                if maybe_mps_semidense_data is None:
                    logger.warning(
                        f"Querying MPS SemiDense data for {timestamps_ns} has returned None, skipping this sample."
                    )
                    return None

                # Fill MPS SemiDense data into sample
                sample.mps_semidense_point_data = maybe_mps_semidense_data

            # =======================================
            # RGB Depth data
            # =======================================
            elif isinstance(processor, DepthImageProcessor):
                maybe_depth_data = processor.get_depth_data_by_timestamps_ns(
                    timestamps_ns
                )
                if maybe_depth_data is None:
                    logger.warning(
                        f"Querying Depth data for {timestamps_ns} has returned None, skipping this sample."
                    )
                    return None

                # Fill depth data into sample
                sample.camera_rgb_depth = maybe_depth_data

            # ========================================
            # GT data
            # ========================================
            elif isinstance(processor, EfmGtProcessor):
                maybe_gt_data = processor.get_gt_by_timestamp_list_ns(timestamps_ns)
                if maybe_gt_data is None:
                    logger.warning(
                        f"Querying GT data for {timestamps_ns} has returned None, skipping this sample."
                    )
                    return None
                sample.gt_data["efm_gt"] = maybe_gt_data

            else:
                raise ValueError(
                    f"Unimplemented processor class {processor.__name__} in SampleBuilder! "
                )

        # only return data of common timestamps among all sub-data types
        if not self._check_timestamps_are_consistent(sample, timestamps_ns):
            logger.warning(
                "Timestamps in sample are not consistent, skipping this sample."
            )
            return None

        return sample

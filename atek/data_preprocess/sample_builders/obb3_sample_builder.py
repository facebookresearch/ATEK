# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Dict, Optional

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsTrajData,
    MultiFrameCameraData,
)
from atek.data_preprocess.processors.aria_camera_processor import AriaCameraProcessor
from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from atek.data_preprocess.processors.obb3_gt_processor import Obb3GtProcessor
from omegaconf.omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ObjectDetection3dSampleBuilder:
    """
    A baseline builder for object detection 3D tasks
    """

    def __init__(
        self,
        conf: DictConfig,
        vrs_file: str,
        mps_files: Optional[Dict[str, str]] = None,
        gt_files: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        vrs_file: the main Aria vrs file
        """
        self.conf = conf

        self.vrs_file = vrs_file

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
        for camera_conf in camera_conf_list:
            if camera_conf.selected:
                processors[camera_conf.sensor_label] = AriaCameraProcessor(
                    vrs_file, camera_conf
                )

        if "mps_traj" in conf and conf.mps_traj.selected:
            processors["mps_traj"] = MpsTrajProcessor(
                mps_closedloop_traj_file=mps_files["mps_closedloop_traj_file"],
                conf=conf.mps_traj,
            )

        if "obb3_gt" in conf and conf.obb3_gt.selected:
            processors["obb3_gt"] = Obb3GtProcessor(
                obb3_file_path=gt_files["obb3_file"],
                obb3_traj_file_path=gt_files["obb3_traj_file"],
                instance_json_file_path=gt_files["instance_json_file"],
                category_mapping_file_path=gt_files.get(
                    "category_mapping_file", None
                ),  # this file is optional
                conf=conf.obb3_gt,
            )

        return processors

    def get_sample_by_timestamp_ns(self, timestamp_ns) -> Optional[AtekDataSample]:
        sample = AtekDataSample()

        for processor_label, processor in self.processors.items():
            if isinstance(processor, AriaCameraProcessor):
                # ========================================
                # Aria camera sensor data
                # ========================================
                maybe_image_data = processor.get_image_data_by_timestamp_ns(
                    timestamp_ns
                )
                # Skip if no image data is available
                if maybe_image_data is None:
                    logger.warning(
                        f"Querying camera for {timestamp_ns} on processor {processor_label} has returned None, skipping this sample."
                    )
                    return None

                # Fill image data into sample
                sample_camera_data = MultiFrameCameraData()
                sample_camera_data.images = maybe_image_data[0]
                sample_camera_data.capture_timestamps_ns = maybe_image_data[1]
                sample_camera_data.frame_ids = maybe_image_data[2]

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
                maybe_mps_traj_data = processor.get_closed_loop_pose_by_timestamp_ns(
                    timestamp_ns
                )
                if maybe_mps_traj_data is None:
                    logger.warning(
                        f"Querying MPS traj for {timestamp_ns} has returned None, skipping this sample."
                    )
                    return None

                # Fill MPS traj data into sample
                sample_traj_data = MpsTrajData()
                # [3,4] -> [n_frames, 3, 4]
                sample_traj_data.Ts_World_Device = maybe_mps_traj_data[0]
                sample_traj_data.capture_timestamps_ns = maybe_mps_traj_data[1]
                sample_traj_data.gravity_in_world = maybe_mps_traj_data[2]

                sample.mps_traj_data = sample_traj_data
            # ========================================
            # GT data
            # ========================================
            elif isinstance(processor, Obb3GtProcessor):
                maybe_gt_data = processor.get_gt_by_timestamp_ns(timestamp_ns)
                if maybe_gt_data is None:
                    logger.warning(
                        f"Querying GT data for {timestamp_ns} has returned None, skipping this sample."
                    )
                    return None
                sample.gt_data = maybe_gt_data

            else:
                raise ValueError(
                    f"Unimplemented processor class {processor.__name__} in SampleBuilder! "
                )

        return sample

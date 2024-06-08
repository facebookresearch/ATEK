# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Dict, List, Optional

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
)
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
    A Sample builder for EFM, which performs Obb3 detection and surface recon task.
    """

    def __init__(
        self,
        conf: DictConfig,
        vrs_file: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
        depth_vrs_file: str,
    ) -> None:
        """
        vrs_file: the main Aria vrs file
        """
        self.conf = conf

        self.vrs_file = vrs_file

        self.processors = self._add_processors_from_conf(
            conf, vrs_file, mps_files, gt_files, depth_vrs_file
        )

    def _add_processors_from_conf(
        self,
        conf: DictConfig,
        vrs_file: str,
        mps_files: Dict[str, str],
        gt_files: Dict[str, str],
        depth_vrs_file: str,
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
        if "rgb-depth" in conf and conf.rgb_depth.selected:
            assert (
                depth_vrs_file is not None
            ), "need to specify depth vrs file to use depth processor"

            # Obtain image transformations from rgb AriaCameraProcessor, where interpolation needs to be exactly set to NEAREST
            assert (
                "camera-rgb" in processors
            ), "rgb_depth depends on camera_rgb processor to obtain camera calibration"
            depth_image_transform_list = processors[
                "camera-rgb"
            ].get_image_transform_list(rescale_interpolation=InterpolationMode.NEAREST)

            processors["rgb_depth"] = DepthImageProcessor(
                depth_vrs=depth_vrs_file,
                image_transform_list=depth_image_transform_list,
                conf=conf.rgb_depth,
            )

        if "efm_gt" in conf and conf.efm_gt.selected:
            processors["efm_gt"] = EfmGtProcessor(
                obb3_file_path=gt_files["obb3_file"],
                obb3_traj_file_path=gt_files["obb3_traj_file"],
                instance_json_file_path=gt_files["instance_json_file"],
                category_mapping_file_path=gt_files.get(
                    "category_mapping_file", None
                ),  # this file is optional
                conf=conf.efm_gt,
            )

        return processors

    def get_sample_by_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[AtekDataSample]:
        sample = AtekDataSample()

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
                sample.mps_semidense_point_data = MpsSemiDensePointData(
                    points_world=maybe_mps_semidense_data[0],
                    points_inv_dist_std=maybe_mps_semidense_data[1],
                )

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
                sample.gt_data = maybe_gt_data

            else:
                raise ValueError(
                    f"Unimplemented processor class {processor.__name__} in SampleBuilder! "
                )

        return sample
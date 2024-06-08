# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, List, Optional

import torch
from atek.data_preprocess.util.file_io_utils import concat_list_of_tensors


@dataclass
class MultiFrameCameraData:
    """
    A class to store multiple frames from Aria camera stream
    """

    # multiple frames where K is the number of frames
    images: torch.Tensor = None  # [num_frames, num_channels, width, height]
    capture_timestamps_ns: torch.Tensor = None  # [num_frames]
    frame_ids: torch.Tensor = None  # [num_frames]
    exposure_durations_s: torch.Tensor = None  # [num_frames]
    gains: torch.Tensor = None  # [num_frames]

    # calibration params that are the same for all frames
    camera_label: str = ""
    T_Device_Camera: torch.Tensor = None  # [num_frames, 3, 4], R|t
    camera_model_name: str = ""
    projection_params: torch.Tensor = None  # intrinsics
    camera_valid_radius: torch.Tensor = None  # [1]
    origin_camera_label: str = ""  # camera label of the "Device" frame

    @staticmethod
    def image_field_names():
        return ["images"]

    @staticmethod
    def tensor_field_names():
        return [
            "capture_timestamps_ns",
            "frame_ids",
            "exposure_durations_s",
            "gains",
            "T_Device_Camera",
            "projection_params",
            "camera_valid_radius",
        ]

    @staticmethod
    def str_field_names():
        return [
            "camera_label",
            "camera_model_name",
            "origin_camera_label",
        ]

    def to_flatten_dict(self):
        """
        Transforms to a flattened dictionary, excluding attributes with None values.
        Attributes are prefixed to ensure uniqueness and to maintain context.
        """
        flatten_dict = {}
        for f in fields(self):
            field_name = f.name
            # Skip if field is None or empty string
            if (getattr(self, field_name) is None) or (getattr(self, field_name) == ""):
                continue

            # Process images separately
            if field_name in self.image_field_names():
                # Transpose dimensions
                image_frames_in_np = (
                    getattr(self, field_name).numpy().transpose(0, 2, 3, 1)
                )

                for id, img in enumerate(image_frames_in_np):
                    flatten_dict[f"MFCD#{self.camera_label}+images_{id}.jpeg"] = (
                        img if img.shape[-1] == 3 else img.squeeze()
                    )
                continue

            # add file extentions so that WDS writer knows how to handle the data
            if field_name in self.tensor_field_names():
                file_extension = ".pth"
            elif field_name in self.str_field_names():
                file_extension = ".txt"
            else:
                file_extension = ".json"

            flatten_dict[f"MFCD#{self.camera_label}+{field_name}{file_extension}"] = (
                getattr(self, field_name)
            )

        return flatten_dict


@dataclass
class ImuData:
    """
    A class to store data from IMU stream
    """

    # raw imu data
    raw_accel_data: torch.Tensor = None  # [num_imu_data, 3]
    raw_gyro_data: torch.Tensor = None  # [num_imu_data, 3]
    capture_timestamps_ns: torch.Tensor = None  # [num_frames]

    # rectified imu data
    rectified_accel_data: torch.Tensor = None  # [num_imu_data, 3]
    rectified_gyro_data: torch.Tensor = None  # [num_imu_data, 3]

    # calibration
    imu_label: str = ""
    T_Device_Imu: torch.Tensor = None  # [num_frames, 3, 4], R|t
    accel_rect_matrix: torch.Tensor = None  #  [3x3]
    accel_rect_bias: torch.Tensor = None  # [3]
    gyro_rect_matrix: torch.Tensor = None  # [3x3]
    gyro_rect_bias: torch.Tensor = None  # [3]

    @staticmethod
    def all_field_names():
        return [
            "raw_accel_data",
            "raw_gyro_data",
            "capture_timestamps_ns",
            "rectified_accel_data",
            "rectified_gyro_data",
            "imu_label",
            "T_Device_Imu",
            "accel_rect_matrix",
            "accel_rect_bias",
            "gyro_rect_matrix",
            "gyro_rect_bias",
        ]


@dataclass
class MpsTrajData:
    Ts_World_Device: torch.Tensor = None  # [num_frames, 3, 4], R|t
    capture_timestamps_ns: torch.Tensor = None  # [num_frames,]
    gravity_in_world: torch.Tensor = None  # [3]

    @staticmethod
    def tensor_field_names():
        return [
            "Ts_World_Device",
            "capture_timestamps_ns",
            "gravity_in_world",
        ]

    def to_flatten_dict(self):
        """
        Transforms to a flattened dictionary, excluding attributes with None values.
        Attributes are prefixed to ensure uniqueness and to maintain context.
        """
        flatten_dict = {}
        for f in fields(self):
            field_name = f.name
            # Skip if field is None or empty string
            if (getattr(self, field_name) is None) or (getattr(self, field_name) == ""):
                continue

            # add file extentions so that WDS writer knows how to handle the data
            if field_name in self.tensor_field_names():
                file_extension = ".pth"
            else:
                file_extension = ".json"

            flatten_dict[f"MTD#{field_name}{file_extension}"] = getattr(
                self, field_name
            )

        return flatten_dict


@dataclass
class MpsSemiDensePointData:
    points_world: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N, 3] to represent observable points, List has length of num_frames
    points_inv_dist_std: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N] to represent points' inverse distance, List has length of num_frames

    def to_flatten_dict(self):
        """
        Transforms to a flattened dictionary, excluding attributes with None values.
        Attributes are prefixed to ensure uniqueness and to maintain context.
        Semidense point data needs to be flattended from List[Tensor (N, 3)] to a Tensor (M, 3) in order to be writable to WDS.
        Therefore we store a `stacked_points_world` (Tensor [M, 3]) along with `points_world_lengths` (Tensor [num_frames]),
        in order to unpack the stacked tensor later. Same for `points_inv_dist_std`.
        """
        flatten_dict = {}

        # obtain the "lengths" of each tensor in list.
        stacked_points_world, len_points_world = concat_list_of_tensors(
            self.points_world
        )
        stacked_points_inv_dist, len_points_inv_dist = concat_list_of_tensors(
            self.points_inv_dist_std
        )

        assert torch.allclose(
            len_points_world, len_points_inv_dist, atol=1
        ), f"The lengths of points_world and points_inv_dist should be the same! Instead got\n {len_points_world} vs \n {len_points_inv_dist}"

        # add to flatten dict
        flatten_dict[f"MSDPD#points_world_lengths.pth"] = len_points_world
        flatten_dict[f"MSDPD#stacked_points_world.pth"] = stacked_points_world
        flatten_dict[f"MSDPD#stacked_points_inv_dist_std.pth"] = stacked_points_inv_dist

        return flatten_dict


@dataclass
class AtekDataSample:
    """
    Underlying data structure for ATEK data sample.
    """

    # Aria sensor data
    camera_rgb: Optional[MultiFrameCameraData] = None
    camera_slam_left: Optional[MultiFrameCameraData] = None
    camera_slam_right: Optional[MultiFrameCameraData] = None
    # camera_et_left: Optional[MultiFrameCameraData] = None
    # camera_et_right: Optional[MultiFrameCameraData] = None
    # imu_left: Optional[ImuData] = None

    # MPS data
    mps_traj_data: Optional[MpsTrajData] = None
    mps_semidense_point_data: Optional[MpsSemiDensePointData] = None

    # Depth data
    camera_rgb_depth: Optional[MultiFrameCameraData] = None

    # GT data, represented by a dictionary
    gt_data: Dict = field(default_factory=dict)

    def to_flatten_dict(self):
        flatten_dict = {}
        for field_name, field_value in self.__dict__.items():
            # Skip if the field value is None
            if field_value is None:
                continue
            # update with flatten sub-dataclasses
            if is_dataclass(field_value) and hasattr(field_value, "to_flatten_dict"):
                flatten_dict.update(field_value.to_flatten_dict())
            # rename gt_data
            elif field_name == "gt_data":
                flatten_dict["GtData.json"] = field_value
            else:
                raise ValueError(f"This field {field_name} is not implemented yet!")
        return flatten_dict

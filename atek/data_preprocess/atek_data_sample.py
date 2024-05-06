# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class MultiFrameCameraData:
    """
    A class to store multiple frames from Aria camera stream
    """

    # multiple frames where K is the number of frames
    images: torch.Tensor = None  # [num_frames, num_channels, width, height]
    capture_timestamps_ns: torch.Tensor = None  # [num_frames]
    frame_ids: torch.Tensor = None  # [num_frames]

    # calibration params that are the same for all frames
    camera_label: str = ""
    T_Device_Camera: torch.Tensor = None  # [num_frames, 3, 4], R|t
    camera_model_name: str = ""
    projection_params: torch.Tensor = None  # intrinsics
    origin_camera_label: str = ""  # camera label of the "Device" frame

    @staticmethod
    def all_field_names():
        return [
            "images",
            "capture_timestamps_ns",
            "frame_ids",
            "camera_label",
            "T_Device_Camera",
            "camera_model_name",
            "camera_params",
            "origin_camera_label",
        ]


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
    def all_field_names():
        return [
            "Ts_World_Device",
            "capture_timestamps_ns",
            "gravity_in_world",
        ]


@dataclass
class MpsSemidensePointData:
    points_world: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N, 3] to represent observable points, List has length of num_frames
    points_inv_dist_std: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N] to represent points' inverse distance, List has length of num_frames

    @staticmethod
    def all_field_names():
        return [
            "points_world",
            "points_inv_dist_std",
        ]


@dataclass
class AtekDataSample:
    """
    Underlying data structure for ATEK data sample.
    """

    # Aria sensor data
    camera_rgb: Optional[MultiFrameCameraData] = None
    camera_slam_left: Optional[MultiFrameCameraData] = None
    camera_slam_right: Optional[MultiFrameCameraData] = None
    camera_et_left: Optional[MultiFrameCameraData] = None
    camera_et_right: Optional[MultiFrameCameraData] = None
    imu_left: Optional[ImuData] = None

    # MPS data
    mps_traj_data: Optional[MpsTrajData] = None
    # TODO: add this: mps_semidense_point_data: Optional[MpsSemidensePointData] = None

    # GT data, represented by a dictionary
    gt_data: Dict = field(default_factory=dict)

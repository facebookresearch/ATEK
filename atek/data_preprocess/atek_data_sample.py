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

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Optional

import torch
from atek.util.tensor_utils import concat_list_of_tensors


def _to_flatten_dict_impl(dataclass_instance, dataclass_prefix):
    flatten_dict = {}
    # asdict will create a deep copy. TODO: is this necessary?
    for (
        key,
        value,
    ) in asdict(dataclass_instance).items():
        if value is None or value == "":
            continue

        # add prefix to the key
        flattened_key_in_lcase = f"{dataclass_prefix}{key}".lower()
        flatten_dict[flattened_key_in_lcase] = value
    return flatten_dict


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
        Attributes are prefixed to ensure uniqueness and to maintain context. Keys are lower-cased to be consistent with WDS tariterator behavior
        """
        # mfcd stands for MultiFrameCameraData, and need to further append camera_label to the prefix, so that the flattened key looks like "mfcd#camera-rgb+images".
        return _to_flatten_dict_impl(
            dataclass_instance=self, dataclass_prefix=f"mfcd#{self.camera_label}+"
        )


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

    def to_flatten_dict(self):
        # imud stands for ImuData
        return _to_flatten_dict_impl(dataclass_instance=self, dataclass_prefix="imud#")


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
        # mtd stands for MpsTrajData
        return _to_flatten_dict_impl(dataclass_instance=self, dataclass_prefix="mtd#")


@dataclass
class MpsSemiDensePointData:
    points_world: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N, 3] to represent observable points, List has length of num_frames
    points_dist_std: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N] to represent points' distance, List has length of num_frames
    points_inv_dist_std: List[torch.Tensor] = field(
        default_factory=list
    )  # Tensor has shape of [N] to represent points' inverse distance, List has length of num_frames
    capture_timestamps_ns: torch.Tensor = None  # [num_frames]
    points_volumn_min: torch.Tensor = None  # [3], xyz
    points_volumn_max: torch.Tensor = None  # [3], xyz

    def to_flatten_dict(self):
        # "msdpd" stands for MpsSemiDensePointData
        return _to_flatten_dict_impl(dataclass_instance=self, dataclass_prefix="msdpd#")


@dataclass
class MpsOnlineCalibData:
    capture_timestamps_ns: Optional[torch.Tensor] = None  # [num_frames]
    utc_timestamps_ns: Optional[torch.Tensor] = None  # [num_frames]

    # camera calibration, [num_timestamps, num_of_camera, number_of_params]
    # now all cameras have same number of params(15), but in the future, if
    # we have different number of params for different cameras, we can use a list of tensors
    # to store the calibration params for each camera.

    projection_params: Optional[torch.Tensor] = None
    # online_calib_camera_labels: List[str] todo: add camera labels to online calib data in future
    #  TODO to support varying intrinsics param count. Filed task: T196065139

    ts_device_camera: Optional[torch.Tensor] = (
        None  # Tensor has shape of [num_timestamps, num_of_camera, 3, 4]
    )

    # TODO: we can add calibration for IMU in the future

    def to_flatten_dict(self):
        # mocd stands for MpsOnlineCalibData
        return _to_flatten_dict_impl(dataclass_instance=self, dataclass_prefix="mocd#")


@dataclass
class AtekDataSample:
    """
    Underlying data structure for ATEK data sample.
    """

    # name of the sequence
    sequence_name: Optional[str] = None

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
    mps_online_calib_data: Optional[MpsOnlineCalibData] = None

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

            if field_name in ["gt_data", "sequence_name"]:
                flatten_dict[field_name] = field_value
                continue

            # update with flatten sub-dataclasses
            if is_dataclass(field_value) and hasattr(field_value, "to_flatten_dict"):
                flatten_dict.update(field_value.to_flatten_dict())
            else:
                raise ValueError(
                    f"This field {field_name} does not have flatten_to_dict implemented yet!"
                )
        return flatten_dict


def _init_data_class_from_flatten_dict_impl(
    DataClassType, flatten_dict: Dict[str, Any], prefix: str = ""
):
    """
    A helper (impl) to initialize a data class from its corresponding flattened dictionary.
    """
    # find fields in flatten dict that starts with prefix, create a sub-dict from it, with the prefix removed from the keys
    sub_dict = {}
    for key in flatten_dict.keys():
        if key.startswith(prefix):
            sub_dict[key[len(prefix) :]] = flatten_dict[key]
    if len(sub_dict) == 0:
        return None

    data_class_instance = DataClassType()

    # Populate the dataclass fields
    for data_field in fields(data_class_instance):
        data_field_name = data_field.name
        if data_field_name.lower() in sub_dict:
            setattr(
                data_class_instance,
                data_field_name,
                sub_dict[data_field_name.lower()],
            )
        else:
            continue

    return data_class_instance


def create_atek_data_sample_from_flatten_dict(flatten_dict):
    """
    A helper function to initialize an ATEK data sample from its corresponding flattened dictionary.
    """
    # Camera data + depth
    atek_data_sample = AtekDataSample()
    atek_data_sample.camera_rgb = _init_data_class_from_flatten_dict_impl(
        MultiFrameCameraData,
        flatten_dict,
        "mfcd#camera-rgb+",
    )
    atek_data_sample.camera_slam_left = _init_data_class_from_flatten_dict_impl(
        MultiFrameCameraData,
        flatten_dict,
        "mfcd#camera-slam-left+",
    )
    atek_data_sample.camera_slam_right = _init_data_class_from_flatten_dict_impl(
        MultiFrameCameraData,
        flatten_dict,
        "mfcd#camera-slam-right+",
    )
    atek_data_sample.camera_rgb_depth = _init_data_class_from_flatten_dict_impl(
        MultiFrameCameraData,
        flatten_dict,
        "mfcd#camera-rgb-depth+",
    )

    # MPS data
    atek_data_sample.mps_traj_data = _init_data_class_from_flatten_dict_impl(
        MpsTrajData, flatten_dict, "mtd#"
    )
    atek_data_sample.mps_semidense_point_data = _init_data_class_from_flatten_dict_impl(
        MpsSemiDensePointData,
        flatten_dict,
        "msdpd#",
    )
    atek_data_sample.mps_online_calib_data = _init_data_class_from_flatten_dict_impl(
        MpsOnlineCalibData,
        flatten_dict,
        "mocd#",
    )

    # GT data is already a dict
    atek_data_sample.gt_data = flatten_dict["gt_data"]
    atek_data_sample.sequence_name = flatten_dict["sequence_name"]

    return atek_data_sample

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

from typing import List, Optional

import torch
from atek.data_preprocess.atek_data_sample import MpsOnlineCalibData
from omegaconf.omegaconf import DictConfig
from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeQueryOptions  # @manual


class MpsOnlineCalibProcessor:
    """
    processor to handle the MPS online calibration data
    """

    def __init__(
        self,
        mps_online_calib_data_file_path: str,
        conf: DictConfig,
    ):
        self.conf = conf

        mps_data_paths = mps.MpsDataPaths()
        mps_data_paths.slam.online_calibrations = mps_online_calib_data_file_path
        self.mps_data_provider = mps.MpsDataProvider(mps_data_paths)

    def get_online_calibration_by_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[MpsOnlineCalibData]:
        """
        This function is to get online calibration data by a list of timestamps_ns
        """

        projection_params_list = []  # This will be a list of tensors
        t_device_camera_list = []  # This will be a list of tensors
        capture_timestamp_list = []  # This will store capture timestamps
        utc_timestamp_list = []  # This will store UTC timestamps

        for single_timestamp in timestamps_ns:
            online_calib = self.mps_data_provider.get_online_calibration(
                device_timestamp_ns=single_timestamp,
                time_query_options=TimeQueryOptions.CLOSEST,
            )

            if online_calib is None:
                continue

            capture_timestamps_ns = int(
                online_calib.tracking_timestamp.total_seconds() * 1_000_000_000
            )
            if abs(capture_timestamps_ns - single_timestamp) > self.conf.tolerance_ns:
                continue
            capture_timestamp_list.append(capture_timestamps_ns)
            utc_timestamp_list.append(
                int(online_calib.utc_timestamp.total_seconds() * 1_000_000_000)
            )
            projection_params_per_timestamp_list = []
            t_device_camera_per_timestamp_list = []
            camera_calibs = online_calib.camera_calibs

            for camera_calib in camera_calibs:

                projection_params = camera_calib.projection_params()
                projection_params_per_timestamp_list.append(projection_params)

                t_device_camera_per_timestamp_list.append(
                    camera_calib.get_transform_device_camera().to_matrix3x4().tolist()
                )

            projection_params_list.append(
                torch.tensor(projection_params_per_timestamp_list, dtype=torch.float32)
            )
            t_device_camera_list.append(
                torch.tensor(t_device_camera_per_timestamp_list, dtype=torch.float32)
            )

        if len(projection_params_list) == 0:
            return None

        projection_params_tensor = torch.stack(projection_params_list)
        ts_device_camera_tensor = torch.stack(t_device_camera_list)
        capture_timestamp_tensor = torch.tensor(
            capture_timestamp_list, dtype=torch.int64
        )
        utc_timestamp_tensor = torch.tensor(utc_timestamp_list, dtype=torch.int64)

        return MpsOnlineCalibData(
            capture_timestamps_ns=capture_timestamp_tensor,
            utc_timestamps_ns=utc_timestamp_tensor,
            projection_params=projection_params_tensor,
            ts_device_camera=ts_device_camera_tensor,
        )

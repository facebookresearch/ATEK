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
from typing import List, Optional, Tuple

import numpy as np

import torch

from omegaconf.omegaconf import DictConfig

from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeQueryOptions  # @manual

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MpsTrajProcessor:
    def __init__(
        self,
        mps_closedloop_traj_file: str,
        conf: DictConfig,
    ):
        # Parse in conf
        self.conf = conf

        # Create MPS data provider
        mps_data_paths = mps.MpsDataPaths()
        mps_data_paths.slam.closed_loop_trajectory = mps_closedloop_traj_file
        self.mps_data_provider = mps.MpsDataProvider(mps_data_paths)

    def get_closed_loop_pose_by_timestamps_ns(
        self, timestamps_ns: List[int], interpolate: bool = False
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Obtain a single MPS trajectory data by timestamp.
        returns: if successful, returns (T_world_device: Tensor [Frames, 3, 4], R|t, capture_timestamp: Tensor[Frames,], gravity_in_world: Tensor[3,])
                else returns None
        """
        pose_list = []
        capture_timestamp_list = []
        gravity_in_world = None
        for single_timestamp in timestamps_ns:
            if interpolate:
                pose = self.mps_data_provider.getInterpolatedClosedLoopPose(
                    deviceTimeStampNs=single_timestamp,
                )
            else:
                pose = self.mps_data_provider.get_closed_loop_pose(
                    device_timestamp_ns=single_timestamp,
                    time_query_options=TimeQueryOptions.CLOSEST,
                )

            # Check if fetched data is within tolerance. Note that pose tracking timestamp is in us
            capture_timestamp = int(
                pose.tracking_timestamp.total_seconds() * 1_000_000_000
            )
            if abs(capture_timestamp - single_timestamp) > self.conf.tolerance_ns:
                continue

            # Pose tensor [3,4]
            pose_list.append(
                torch.from_numpy(
                    pose.transform_world_device.to_matrix3x4().astype(np.float32)
                )
            )
            capture_timestamp_list.append(capture_timestamp)

            # Gravity only needs to be assigned once
            if gravity_in_world is None:
                gravity_in_world = torch.from_numpy(
                    pose.gravity_world.astype(np.float32)
                )

        # Skip if empty data
        if len(pose_list) == 0:
            return None

        return (
            torch.stack(pose_list, dim=0),
            torch.tensor(capture_timestamp_list, dtype=torch.int64),
            gravity_in_world,
        )

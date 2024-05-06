# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Optional, Tuple

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

    def get_closed_loop_pose_by_timestamp_ns(
        self, timestamp_ns: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Obtain a single MPS trajectory data by timestamp.
        returns: if successful, returns (T_world_device: Tensor [1, 3, 4], R|t, capture_timestamp: Tensor[1,], gravity_in_world: Tensor[3,])
                else returns None
        """
        pose = self.mps_data_provider.get_closed_loop_pose(
            device_timestamp_ns=timestamp_ns,
            time_query_options=TimeQueryOptions.CLOSEST,
        )

        # Check if fetched data is within tolerance. Note that pose tracking timestamp is in us
        capture_timestamp = int(pose.tracking_timestamp.total_seconds() * 1_000_000_000)
        if abs(capture_timestamp - timestamp_ns) > self.conf.tolerance_ns:
            return None

        # properly clean output to desired dtype and shapes
        torch_T_world_device = torch.from_numpy(
            pose.transform_world_device.to_matrix3x4().astype(np.float32)
        )
        torch_T_world_device = torch.unsqueeze(torch_T_world_device, dim=0)
        torch_capture_timestamp = torch.tensor([capture_timestamp], dtype=torch.int64)
        torch_gravity_in_world = torch.from_numpy(pose.gravity_world.astype(np.float32))

        return torch_T_world_device, torch_capture_timestamp, torch_gravity_in_world

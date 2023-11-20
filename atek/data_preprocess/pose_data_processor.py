# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging

from typing import List, Union

import numpy as np

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PoseDataProcessor:
    def __init__(
        self,
        name: str,
        trajectory_file: str,
    ):
        self.name = name
        self.trajectory_file = trajectory_file
        self.traj_df = pd.read_csv(self.trajectory_file)
        self.traj_df = self.traj_df.sort_values("tracking_timestamp_us")
        assert (
            self.traj_df["graph_uid"].nunique() == 1
        ), f"Seq({name})'s trajectory file contains multiple graphs."

        deltas_us = np.diff(self.traj_df["tracking_timestamp_us"])
        self.rate_hz = 1 / np.mean(deltas_us / 1000_000)

    def get_timestamps_ns(self):
        return self.traj_df["tracking_timestamp_us"].values * 1000

    def get_rate_hz(self):
        return self.rate_hz

    def get_nearest_poses(
        self,
        timestamps_ns: Union[np.ndarray, List[int]],
        tolerance_ns: int = 150_000,
        only_return_valid: bool = False,
    ):
        """
        Get the nearest poses to a list of timestamps.
        Args:
            timestamp_ns: A list of timestamps in nanoseconds.
            tolerance_ns: The tolerance for finding the nearest pose.
                If the difference between the input timestamp and the pose timestamp
                is less than this value, then it will be considered as the same pose.
        Returns:
            A list of translation and quaternion with the same length as `timestamps`.
                Translation from camera to world coordinate frame. 3x1
                Quaternion(xyzw) from camera to world coordinate frame. 4x1
                Gravity in world coordinate frame. 3x1
        """
        if isinstance(timestamps_ns, List):
            timestamps_ns = np.array(timestamps_ns)

        timestamps_us_df = pd.DataFrame(
            {
                "tracking_timestamp_us": pd.Series(
                    np.round(timestamps_ns / 1000).astype(int)
                )
            }
        )
        timestamps_us_df = timestamps_us_df.sort_values("tracking_timestamp_us")

        merged_df = pd.merge_asof(
            timestamps_us_df,
            self.traj_df,
            on="tracking_timestamp_us",
            tolerance=int(round(tolerance_ns / 1000)),
            direction="nearest",
        )

        valid_merged_df = merged_df.dropna()

        invalid_count = len(merged_df) - len(valid_merged_df)
        invalid_percent = (invalid_count / len(merged_df)) * 100
        if invalid_percent > 5:
            logger.warning(
                f"{invalid_count} ({invalid_percent:.2f}%) of the input timestamps can not find corresponding poses."
            )

        if only_return_valid:
            merged_df = valid_merged_df

        return merged_df[
            [
                "tx_world_device",
                "ty_world_device",
                "tz_world_device",
                "qx_world_device",
                "qy_world_device",
                "qz_world_device",
                "qw_world_device",
                "gravity_x_world",
                "gravity_y_world",
                "gravity_z_world",
            ]
        ]

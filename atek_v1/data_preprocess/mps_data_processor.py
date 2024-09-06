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

from collections import defaultdict

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MpsDataProcessor:
    def __init__(
        self,
        name: str,
        trajectory_file: str,
        semidense_global_point_file: Optional[str] = None,
        semidense_observation_file: Optional[str] = None,
        semidense_compression_method: Optional[str] = None,
    ):
        self.name = name
        # load in trajectory file
        self.trajectory_file = trajectory_file
        self.traj_df = pd.read_csv(self.trajectory_file)
        self.traj_df = self.traj_df.sort_values("tracking_timestamp_us")
        assert (
            self.traj_df["graph_uid"].nunique() == 1
        ), f"Seq({name})'s trajectory file contains multiple graphs."

        deltas_us = np.diff(self.traj_df["tracking_timestamp_us"])
        self.rate_hz = 1 / np.mean(deltas_us / 1000_000)

        # load in semi-dense point files if available, the results are stored as maps for quick querying by timestamp
        (self.uid_to_p3, self.uid_to_inv_dist_std) = self._load_semidense_global_points(
            semidense_global_point_file, compression_method=semidense_compression_method
        )
        (self.time_to_uids, self.uid_to_times) = self._load_semidense_observations(
            semidense_observation_file, compression_method=semidense_compression_method
        )

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
        matched_df = self._find_matching_timestamps_in_df(
            self.traj_df,
            timestamps_ns,
            tolerance_ns,
            only_return_valid,
            timestamp_key_in_df="tracking_timestamp_us",
        )

        return matched_df[
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

    def get_nearest_semidense_points(
        self,
        timestamps_ns: Union[np.ndarray, List[int]],
        tolerance_ns: int = 150_000,
        only_return_valid: bool = False,
    ) -> pd.DataFrame:
        """
        Get points_in_world close to timestamps.
        Args:
            timestamp_ns: A list of timestamps in nanoseconds, Kx1
            tolerance_ns: The tolerance for finding the nearest frame.
                If the difference between the query timestamp and the semidense point timestamp
                is less than this value, then it will be considered as the same frame.
        Returns:
            a list of Tensors of Nx3 to represent observable world points,
            and a list of Tensors of Nx1 to represent the standard deviation of the semidense points.
        """
        # create data frame to re-use timestamp matching function
        time_to_uids_df = pd.DataFrame(
            list(self.time_to_uids.items()), columns=["tracking_timestamp_us", "uids"]
        )

        matched_time_to_uids_df = self._find_matching_timestamps_in_df(
            time_to_uids_df,
            timestamps_ns,
            tolerance_ns,
            only_return_valid,
            timestamp_key_in_df="tracking_timestamp_us",
        )

        df = pd.DataFrame(columns=["points_world", "points_inv_dist_std"])
        # loop over all matched timestamps, and stack point_in_world into a Nx3 tensor
        for uid_list in matched_time_to_uids_df["uids"]:
            points_world_per_timestamp = []
            inv_dist_std_per_timestamp = []
            for uid in uid_list:
                if (uid in self.uid_to_p3) and (uid in self.uid_to_inv_dist_std):
                    points_world_per_timestamp.append(self.uid_to_p3[uid])
                    inv_dist_std_per_timestamp.append(self.uid_to_inv_dist_std[uid])
                else:
                    raise ValueError(
                        f"Point UID {uid} not found in global semidense point file!"
                    )
            # end for uid
            df = df.append(
                {
                    "points_world": torch.stack(points_world_per_timestamp),
                    "points_inv_dist_std": torch.tensor(inv_dist_std_per_timestamp),
                },
                ignore_index=True,
            )
        # end for uid_list

        return df

    def _find_matching_timestamps_in_df(
        self,
        data_frame: pd.DataFrame,
        timestamps_ns: Union[np.ndarray, List[int]],
        tolerance_ns: int,
        only_return_valid: bool = False,
        timestamp_key_in_df: str = "tracking_timestamp_us",
    ) -> pd.DataFrame:
        """
        Helper function that given a list of timestamps, find the rows containing matching timestamps in the data frame, with some tolerance.
        Returns:
            a data frame containing the matching rows.
        """
        if isinstance(timestamps_ns, List):
            timestamps_ns = np.array(timestamps_ns)

        timestamps_us_df = pd.DataFrame(
            {timestamp_key_in_df: pd.Series(np.round(timestamps_ns / 1000).astype(int))}
        )
        timestamps_us_df = timestamps_us_df.sort_values(timestamp_key_in_df)

        # Use merge_asof to find the nearest pose for each timestamp.
        merged_df = pd.merge_asof(
            timestamps_us_df,
            data_frame,
            on=timestamp_key_in_df,
            tolerance=int(round(tolerance_ns / 1000)),
            direction="nearest",
        )

        if not only_return_valid:
            return merged_df
        else:
            # Check percentage of invalid timestamps.
            valid_merged_df = merged_df.dropna()

            invalid_count = len(merged_df) - len(valid_merged_df)
            invalid_percent = (invalid_count / len(merged_df)) * 100
            if invalid_percent > 5:
                logger.warning(
                    f"{invalid_count} ({invalid_percent:.2f}%) of the input timestamps can not find corresponding poses."
                )

            return valid_merged_df

    def _load_semidense_global_points(
        self,
        path: str,
        compression_method: Optional[str],
    ):
        print(f"loading global semi-dense points from {path}")
        uid_to_p3 = {}
        uid_to_inv_dist_std = {}

        with open(path, "rb") as f:
            csv_data = pd.read_csv(f, compression=compression_method)

            # select points and uids and return mapping
            uid_pts = csv_data[
                ["uid", "inv_dist_std", "px_world", "py_world", "pz_world"]
            ]

            for row in uid_pts.values:
                uid = int(row[0])
                inv_dist_std = float(row[1])
                p3 = torch.from_numpy(row[2:]).float()
                uid_to_p3[uid] = p3
                uid_to_inv_dist_std[uid] = inv_dist_std

        return uid_to_p3, uid_to_inv_dist_std

    def _load_semidense_observations(
        self, path: str, compression_method: Optional[str] = None
    ):
        """
        Load semidense observations from a csv file, returns two-way mapping between timestamp_in_us and point uids.
        Args:
            path: The path to the csv file.
        Returns:
            A tuple of two dictionaries.
            The first dictionary maps from timestamp to a list of uids.
            The second dictionary maps from uid to a list of timestamps.
        """

        logger.info(f"loading semidense observations from {path}")
        time_to_uids = defaultdict(list)
        uid_to_times = defaultdict(list)

        with open(path, "rb") as f:
            csv = pd.read_csv(f, compression=compression_method)
            csv = csv[["uid", "frame_tracking_timestamp_us"]]
            for row in csv.values:
                uid = int(row[0])
                time_ns = int(row[1])
                time_to_uids[time_ns].append(uid)
                uid_to_times[uid].append(time_ns)
        return time_to_uids, uid_to_times

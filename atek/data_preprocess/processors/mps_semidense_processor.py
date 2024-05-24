# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch

from omegaconf.omegaconf import DictConfig

from projectaria_tools.core import mps
from projectaria_tools.core.sensor_data import TimeQueryOptions  # @manual

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TIMESTAMP_KEY_IN_DF: str = "tracking_timestamp_us"


class MpsSemiDenseProcessor:
    def __init__(
        self,
        mps_semidense_points_file: str,
        mps_semidense_observations_file: str,
        conf: DictConfig,
    ):
        # Parse in conf
        self.conf = conf

        # Load in semidense points data. Not using MPSDataProvider because it is not sufficient.
        self.uid_to_p3, self.uid_to_inv_dist_std = self._load_semidense_global_points(
            mps_semidense_points_file
        )
        self.time_to_uids, self.uid_to_times = self._load_semidense_observations(
            mps_semidense_observations_file
        )

    def get_semidense_points_by_timestamps_ns(
        self,
        timestamps_ns: List[int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Obtain a semidense points data by timestamp, where `points_world` are in meters, and `points_inv_dist_std` are in meter^-1.
        returns: if successful, returns (points_world: List[torch.Tensor (N,3)], points_inv_dist_std: List[torch.Tensor (N)]), where len(List) = number of frames, which is 1
                else returns None
        """
        # create data frame to re-use timestamp matching function
        time_to_uids_df = pd.DataFrame(
            list(self.time_to_uids.items()), columns=["tracking_timestamp_us", "uids"]
        )

        matched_time_to_uids_df = self._find_matching_timestamps_in_df(
            data_frame=time_to_uids_df,
            timestamps_ns=timestamps_ns,
            tolerance_ns=self.conf.tolerance_ns,
            only_return_valid=False,
        )

        points_world_all = []
        inv_dist_std_all = []
        # loop over all matched timestamps, and stack point_in_world into a Nx3 tensor
        for uid_list in matched_time_to_uids_df["uids"]:
            points_world = []
            inv_dist_std = []
            for uid in uid_list:
                if (uid in self.uid_to_p3) and (uid in self.uid_to_inv_dist_std):
                    points_world.append(self.uid_to_p3[uid])
                    inv_dist_std.append(self.uid_to_inv_dist_std[uid])
                else:
                    raise ValueError(
                        f"Point UID {uid} not found in global semidense point file!"
                    )
            # end for uid
            points_world_all.append(torch.stack(points_world, dim=0))
            inv_dist_std_all.append(torch.tensor(inv_dist_std))
        # end for uid_list

        return points_world_all, inv_dist_std_all

    def _load_semidense_global_points(
        self,
        path: str,
    ):
        print(f"loading global semi-dense points from {path}")

        # Determine compression method
        if path.endswith(".csv"):
            compression_method = None
        elif path.endswith(".gz"):
            compression_method = "gzip"
        else:
            raise ValueError(f"Unsupported compression method for {path}")

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
        self,
        path: str,
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

        # Determine compression method
        if path.endswith(".csv"):
            compression_method = None
        elif path.endswith(".gz"):
            compression_method = "gzip"
        else:
            raise ValueError(f"Unsupported compression method for {path}")

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

    def _find_matching_timestamps_in_df(
        self,
        data_frame: pd.DataFrame,
        timestamps_ns: Union[np.ndarray, List[int]],
        tolerance_ns: int,
        only_return_valid: bool = False,
    ) -> pd.DataFrame:
        """
        Helper function that given a list of timestamps, find the rows containing matching timestamps in the data frame, with some tolerance.
        Returns:
            a data frame containing the matching rows.
        """
        if isinstance(timestamps_ns, List):
            timestamps_ns = np.array(timestamps_ns)

        timestamps_us_df = pd.DataFrame(
            {TIMESTAMP_KEY_IN_DF: pd.Series(np.round(timestamps_ns / 1000).astype(int))}
        )
        timestamps_us_df = timestamps_us_df.sort_values(TIMESTAMP_KEY_IN_DF)

        # Use merge_asof to find the nearest pose for each timestamp.
        merged_df = pd.merge_asof(
            timestamps_us_df,
            data_frame,
            on=TIMESTAMP_KEY_IN_DF,
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
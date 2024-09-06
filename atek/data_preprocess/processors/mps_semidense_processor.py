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
import time
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
from atek.data_preprocess.atek_data_sample import MpsSemiDensePointData

from omegaconf.omegaconf import DictConfig

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
        time_0 = time.time()
        self.uid_to_p3, self.uid_to_dist_std, self.uid_to_inv_dist_std = (
            self._load_semidense_global_points(mps_semidense_points_file)
        )
        time_1 = time.time()
        self.time_to_uids, self.uid_to_times = self._load_semidense_observations(
            mps_semidense_observations_file
        )
        time_2 = time.time()

        self._compute_semidense_volume()
        time_3 = time.time()

        logger.info(
            f"loading semidense points takes {time_1-time_0} seconds, observations takes {time_2-time_1} seconds"
            f"and computing semidense volume takes {time_3-time_2} seconds"
        )

    def get_semidense_points_by_timestamps_ns(
        self,
        timestamps_ns: List[int],
    ) -> Optional[MpsSemiDensePointData]:
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
        dist_std_all = []
        inv_dist_std_all = []
        # loop over all matched timestamps, and stack point_in_world into a Nx3 tensor
        for uid_list in matched_time_to_uids_df["uids"]:
            # uid_list can also be a single nan value, indicating empty observations at this timestamp. hence needs to handle this separately
            if isinstance(uid_list, List):
                points_world = []
                dist_std = []
                inv_dist_std = []
                for uid in uid_list:
                    if (
                        (uid in self.uid_to_p3)
                        and (uid in self.uid_to_dist_std)
                        and (uid in self.uid_to_inv_dist_std)
                    ):
                        points_world.append(self.uid_to_p3[uid])
                        dist_std.append(self.uid_to_dist_std[uid])
                        inv_dist_std.append(self.uid_to_inv_dist_std[uid])
                    else:
                        raise ValueError(
                            f"Point UID {uid} not found in global semidense point file!"
                        )
                # Sort points by inv_distance, ascending
                combined = list(zip(inv_dist_std, points_world, dist_std))
                combined = sorted(combined, key=lambda x: x[0])
                inv_dist_std, points_world, dist_std = map(list, zip(*combined))

                # end for uid
                points_world_all.append(torch.stack(points_world, dim=0))
                dist_std_all.append(torch.tensor(dist_std))
                inv_dist_std_all.append(torch.tensor(inv_dist_std))
            else:
                points_world_all.append(torch.full((1, 3), float("nan")))
                dist_std_all.append(torch.tensor([float("nan")]))
                inv_dist_std_all.append(torch.tensor([float("nan")]))
        # end for uid_list

        capture_timestamps_ns = torch.tensor(
            matched_time_to_uids_df[TIMESTAMP_KEY_IN_DF].values * 1e3, dtype=torch.int64
        )

        return MpsSemiDensePointData(
            points_world=points_world_all,
            points_dist_std=dist_std_all,
            points_inv_dist_std=inv_dist_std_all,
            capture_timestamps_ns=capture_timestamps_ns,
            points_volumn_max=self.vol_max,
            points_volumn_min=self.vol_min,
        )

    def _load_semidense_global_points(
        self,
        path: str,
    ):
        logger.info(f"loading global semi-dense points from {path}")

        # Determine compression method
        if path.endswith(".csv"):
            compression_method = None
        elif path.endswith(".gz"):
            compression_method = "gzip"
        else:
            raise ValueError(f"Unsupported compression method for {path}")

        uid_to_p3 = {}
        uid_to_dist_std = {}
        uid_to_inv_dist_std = {}

        with open(path, "rb") as f:
            csv_data = pd.read_csv(f, compression=compression_method)

            # select points and uids and return mapping
            uid_pts = csv_data[
                ["uid", "dist_std", "inv_dist_std", "px_world", "py_world", "pz_world"]
            ]

            for row in uid_pts.values:
                uid = int(row[0])
                dist_std = float(row[1])
                inv_dist_std = float(row[2])
                p3 = torch.from_numpy(row[3:]).float()
                uid_to_p3[uid] = p3
                uid_to_dist_std[uid] = dist_std
                uid_to_inv_dist_std[uid] = inv_dist_std

        return uid_to_p3, uid_to_dist_std, uid_to_inv_dist_std

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

    def _compute_semidense_volume(
        self, gpu_memory_mb=8000, quantiles=[0.001, 0.01, 0.05], voxel_size=0.04
    ):
        """
        compute the bounding volume for TSDF fusion

        gpu_memory: how much memory to be reserved for TSDF fusion, default: 8000MB
        quantiles: the quantiles of the point cloud in each 3d dimension to bound the global space,
        default: [0.001, 0.01, 0.05], will test which quantile is the smallest that can fit the memory
        voxel_size: the voxel size in meters for TSDF fusion, default: 0.04
        """
        # at least hold tsdf_vol, vol_weight, vol_color, vox_coords
        vol_memory = gpu_memory_mb / 4
        # assume float32 for volume dtype, how many voxels `vol_memory` translates to
        max_voxels = vol_memory * 1e6 / 4

        # Aggregate all global points
        all_points = torch.stack(list(self.uid_to_p3.values()))

        for q in quantiles:
            self.vol_min = torch.quantile(all_points, q, dim=0)
            self.vol_max = torch.quantile(all_points, 1 - q, dim=0)
            self.vol_min = self.vol_min.detach().cpu()
            self.vol_max = self.vol_max.detach().cpu()

            vox_dim = (self.vol_max - self.vol_min) / voxel_size
            est_num_voxels = vox_dim[0] * vox_dim[1] * vox_dim[2]
            if est_num_voxels < max_voxels:
                print(f"compute global bounding volume as {q} to {1-q} quantile")
                break
        if est_num_voxels > max_voxels:
            print("Warning: scene volume too large for TSDF fusion")

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_possible_file_conventions() -> Dict[str, List[str]]:
    """
    Get the file location conventions for various datasets:
    - ADT
    - ASE
    """
    return {
        "video_vrs_file": ["video.vrs"],
        # MPS files
        "mps_closedloop_traj_file": ["aria_trajectory.csv"],
        "mps_semidense_points_file": [
            "mps/slam/semidense_points.csv.gz",  # ADT
            "maps/maps_v1/globalcloud_GT.csv.gz",  # ASE
        ],
        "mps_semidense_observations_file": [
            "mps/slam/semidense_observations.csv.gz",  # ADT
            "maps/maps_v1/observations.csv.gz",  # ASE
        ],
        "mps_online_calib_file": [
            "mps/slam/online_calibration.jsonl",
        ],
        # Depth file
        "depth_vrs_file": ["depth_images.vrs"],
        # GT files
        "gt_obb3_file": ["3d_bounding_box.csv"],
        "gt_obb3_traj_file": ["scene_objects.csv"],
        "gt_obb2_file": ["2d_bounding_box.csv"],
        "gt_instance_json_file": ["instances.json"],
    }


class AtekDataPathsProvider:
    """
    A class that searches for the necessary filepaths for ATEK sample builders, given a data root path.
    returns a dict.

    Currently supported Aria open datasets:
    - ADT
    - ASE
    """

    def __init__(
        self,
        data_root_path: str,
    ) -> None:
        self.data_root_path = data_root_path
        self.atek_data_paths = {}

        # Try to locate files with various conventions
        possible_file_locations = _get_possible_file_conventions()

        # Find the files
        for atek_filename, possible_locations in possible_file_locations.items():
            for possible_location in possible_locations:
                full_file_location = os.path.join(
                    self.data_root_path, possible_location
                )
                if os.path.exists(full_file_location):
                    self.atek_data_paths[atek_filename] = full_file_location
                    break

        # For testing only
        logger.info(f"Located ATEK data paths: {self.atek_data_paths}")

    def get_data_paths(self) -> Dict[str, str]:
        return self.atek_data_paths

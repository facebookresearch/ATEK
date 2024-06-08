# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from atek.data_preprocess.util.file_io_utils import load_category_mapping_from_csv

from omegaconf.omegaconf import DictConfig

from projectaria_tools.core.sophus import SE3
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPaths,
    AriaDigitalTwinDataProvider,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ATEK_OTHER_CATETORY_ID: int = (
    0  # 0 is reserved for other categories in ATEK object taxonomy
)


class Obb3GtProcessor:
    """
    Processes ground truth data for 3D object bounding boxes (OBB3), specifically designed for use in machine learning models focused on 3D object detection tasks.
    This class interfaces with Aria Digital Twin data providers to fetch and process 3D bounding box data, transforming raw data into a structured format suitable for training ML models.
    """

    def __init__(
        self,
        obb3_file_path: str,
        obb3_traj_file_path: str,
        instance_json_file_path: str,
        category_mapping_file_path: Optional[str],
        conf: DictConfig,
    ) -> None:
        """
        Initializes the Obb3GtProcessor with paths to data files and configuration.
        Args:
            obb3_file_path (str): Path to the OBB3 file containing bounding box data.
            obb3_traj_file_path (str): Path to the trajectory data for bounding boxes.
            instance_json_file_path (str): Path to the JSON file containing instance metadata.
            category_mapping_file_path (Optional[str]): Path to the CSV file mapping category names to IDs, in the format of:
                {
                    $KEY_TO_MAP： [“cat_name”, category_id],
                    ...
                },
                where "KEY_TO_MAP" is one of strings of {"prototype_name", "category"}, set through conf.category_mapping_field_name
            conf (DictConfig): Configuration object specifying operational parameters such as data field mappings and tolerances, example yaml:
                ```
                selected: true  # whether to use this processor
                tolerance_ns : 10_000_000 # tolerance in ns for timestamp
                category_mapping_field_name: prototype_name # {prototype_name (for ADT), category (for ASE)}
                ```
        Raises:
            FileNotFoundError: If any of the specified files do not exist at the provided paths.
            ValueError: If configuration parameters are invalid or not compatible.

        Initialize the Obb3GtProcessor object.

        Args:
            obb3_file_path (str): The path to the OBB3 file, in ADT format: https://fburl.com/zh0p3egs
            obb3_traj_file_path (str): The path to the OBB3 trajectory file, in ADT format: https://fburl.com/ut0bddnf
            instance_json_file_path (str): The path to the instance JSON file, in ADT format: https://fburl.com/u4te445v
            category_mapping (Dict): The category mapping dictionary
        """
        self.conf = conf

        # Create a ADT data provider object, which will be used to load 3D BBOX GT data
        data_paths = AriaDigitalTwinDataPaths()
        data_paths.object_boundingbox_3d_filepath = obb3_file_path
        data_paths.object_trajectories_filepath = obb3_traj_file_path
        data_paths.instances_filepath = instance_json_file_path

        self.adt_gt_provider = AriaDigitalTwinDataProvider(data_paths)

        self.category_mapping = (
            None
            if category_mapping_file_path is None
            else load_category_mapping_from_csv(category_mapping_file_path)
        )

    def _center_object_bb3d(
        self, aabb: np.ndarray, T_world_bb3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centers the 3D bounding box based on axis-aligned bounding box (AABB) coordinates and world transformation matrix.
        Args:
            aabb (np.ndarray): Array containing the min and max points [xmin, xmax, ymin, ymax, zmin, zmax].
            T_world_bb3d (np.ndarray): Transformation matrix from world coordinates to bounding box coordinates.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the object dimensions and the new world transformation matrix centered at the object.

        Helper function to transform the object coordinate to the object center
        and generate the new T_world_object with new object coordinate
        aabb: [xmin, xmax, ymin, ymax, zmin, zmax]
        """
        object_dimension = aabb[1::2] - aabb[::2]
        t_center_in_object = (aabb[1::2] + aabb[::2]) / 2

        T_bb3d_object_centered = SE3.from_quat_and_translation(
            1, np.array([0, 0, 0]), t_center_in_object
        )
        T_world_object_centered = T_world_bb3d @ T_bb3d_object_centered

        return object_dimension, T_world_object_centered

    def _obtain_obj_category_info(self, instance_id: int) -> Tuple[str, int]:
        """ """
        instance_info = self.adt_gt_provider.get_instance_info_by_id(instance_id)

        if not self.category_mapping:
            # If no category mapping is provided, we use the original category name.
            category_name = instance_info.category
            category_id = instance_info.category_uid
        else:
            # Query the mapping field from instance
            key_to_map = getattr(
                instance_info, self.conf.category_mapping_field_name, None
            )
            if not key_to_map:
                raise ValueError(
                    f'Unsupported instance field to map: {self.conf.category_mapping_field_name}, need to be ["prototype_name" or "category"]'
                )

            # Perform mapping
            if key_to_map in self.category_mapping:
                category_name = self.category_mapping[key_to_map][0]
                category_id = int(self.category_mapping[key_to_map][1])
            else:
                category_name = "other"
                category_id = ATEK_OTHER_CATETORY_ID

        return category_name, category_id

    def get_gt_by_timestamp_ns(self, timestamp_ns: int) -> Optional[Dict]:
        """
        Retrieves the ground truth data for a given timestamp, formatted as a dictionary.
        Args:
            timestamp_ns (int): The timestamp in nanoseconds for which to retrieve the ground truth data.

        Returns:
            Optional[Dict]: A dictionary containing the ground truth data if available and valid; otherwise, None.

            If not None, the returned dictionary will have the following structure:
                {
                    "instance_id": {
                        "instance_id": str,
                        "category_name": str,
                        "category_id": int,
                        "object_dimensions": torch.Tensor (shape: [3], float32),
                        "T_World_Object": torch.Tensor (shape: [3, 4], float32)
                    },
                    ...
                }

            Each key in the dictionary is a unique instance ID corresponding to an object.

        Notes:
            Returns None if the data at the specified timestamp is not valid or does not meet the configured tolerances.
        """
        bbox3d_with_dt = (
            self.adt_gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
                timestamp_ns,
            )
        )
        # no valid data, return empty dict
        if not bbox3d_with_dt.is_valid():
            logger.warn(f"Cannot obtain valid 3d bbox data at {timestamp_ns}")
            return None

        # queried data out of tolerance, return empty dict
        if bbox3d_with_dt.dt_ns() > self.conf.tolerance_ns:
            logger.warn(
                f"Can not get good 3d bounding boxes at {timestamp_ns} because "
                f"the nearest bb2d with delta time {bbox3d_with_dt.dt_ns()}ns "
                f"bigger than the threshold we have {self.conf.tolerance_ns}ns "
            )
            return None

        # pack 3d bbox data into a dict
        bbox3d_dict = {}
        for instance_id, single_data in bbox3d_with_dt.data().items():
            single_bbox3d_dict = {}
            # fill in instance id and category information
            single_bbox3d_dict["instance_id"] = instance_id
            single_bbox3d_dict["category_name"], single_bbox3d_dict["category_id"] = (
                self._obtain_obj_category_info(instance_id)
            )

            # fill in 3d aabb information, need to put the object coorindate to box center
            aabb_non_centered = single_data.aabb
            T_world_object_non_centered = single_data.transform_scene_object
            (object_dimensions, T_world_object) = self._center_object_bb3d(
                aabb_non_centered, T_world_object_non_centered
            )
            # convert to tensor
            single_bbox3d_dict["object_dimensions"] = torch.from_numpy(
                object_dimensions.astype(np.float32)
            )
            single_bbox3d_dict["T_World_Object"] = torch.from_numpy(
                T_world_object.to_matrix3x4().astype(np.float32)
            )

            bbox3d_dict[instance_id] = single_bbox3d_dict

        return bbox3d_dict

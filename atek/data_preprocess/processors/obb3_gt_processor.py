# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch

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
    A Ground truth (GT) processor class for Object bounding box 3D (OBB3) data, used for 3D object detection ML task
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
        Initialize the Obb3GtProcessor object.

        Args:
            obb3_file_path (str): The path to the OBB3 file, in ADT format: https://fburl.com/zh0p3egs
            obb3_traj_file_path (str): The path to the OBB3 trajectory file, in ADT format: https://fburl.com/ut0bddnf
            instance_json_file_path (str): The path to the instance JSON file, in ADT format: https://fburl.com/u4te445v
            category_mapping (Dict): The category mapping dictionary in the format of:
                {
                    "key_to_map"： [“cat_name”, category_id],
                    ...
                },
                where "key_to_map" is one of {"prototype_name", "category"} in the `instance.json` file, set through conf.category_mapping_field_name
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
            else self._load_category_mapping_from_csv(category_mapping_file_path)
        )

    def _load_category_mapping_from_csv(
        self,
        category_mapping_csv_file: str,
    ) -> Dict:
        """
        Load the category mapping from a CSV file.

        Args:
            category_mapping_csv_file (str): The path to the category mapping CSV file.
            The CSV file should contain exactly 3 Columns, representing "old_category_name or prototype_name", "atek_category_name", "atek_category_id".

        Returns:
            Dict: The category mapping dictionary in the format of:
                {
                    "old_cat/prototype_name"： [“cat_name”, category_id],
                    ...
                }
        """
        with open(category_mapping_csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert (
                len(header) == 3
            ), "Expected 3 columns in the category mapping csv file"
            assert (
                header[1] == "ATEK Category Name" and header[2] == "ATEK Category Id"
            ), f"Column names must be  ATEK Category Name and ATEK Category Id, but got {header[1]} and {header[2]} instead."
            category_mapping = {rows[0]: (rows[1], rows[2]) for rows in reader}
        return category_mapping

    def _center_object_bb3d(
        self, aabb: np.ndarray, T_world_bb3d: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
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
        """
        Helper function to obtain the object category name and category id
        """
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
                category_id = self.category_mapping[key_to_map][1]
            else:
                category_name = "other"
                category_id = ATEK_OTHER_CATETORY_ID

        return category_name, category_id

    def get_gt_by_timestamp_ns(self, timestamp_ns: int) -> Optional[Dict]:
        """
        get obb3 GT by timestamp in nanoseconds
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
        for instance_id, bbox3d_data in bbox3d_with_dt.data().items():
            single_bbox3d_dict = {}
            # fill in instance id and category information
            single_bbox3d_dict["instance_id"] = instance_id
            single_bbox3d_dict["category_name"], single_bbox3d_dict["category_id"] = (
                self._obtain_obj_category_info(instance_id)
            )

            # fill in 3d aabb information, need to put the object coorindate to box center
            single_data = bbox3d_with_dt.data()[instance_id]
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

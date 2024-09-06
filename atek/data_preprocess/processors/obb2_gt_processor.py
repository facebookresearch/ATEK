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
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from atek.util.file_io_utils import load_category_mapping_from_csv

from omegaconf.omegaconf import DictConfig
from projectaria_tools.core.calibration import CameraCalibration

from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPaths,
    AriaDigitalTwinDataProvider,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ATEK_OTHER_CATETORY_ID: int = (
    0  # 0 is reserved for other categories in ATEK object taxonomy
)


class Obb2GtProcessor:
    """
    A Ground truth (GT) processor class for Object bounding box 2D (OBB2) data, used for 2D object detection ML task
    """

    def __init__(
        self,
        obb2_file_path: str,
        instance_json_file_path: str,
        category_mapping_file_path: Optional[str],
        camera_label_to_stream_ids: Dict[str, str],
        camera_label_to_pixel_transforms: Optional[Dict[str, Callable]],
        camera_label_to_calib: Optional[Dict[str, CameraCalibration]],
        conf: DictConfig,
    ) -> None:
        """
        Initializes the Obb2GtProcessor object with necessary file paths and configurations.

        Args:
            obb2_file_path (str): The file path to the OBB2 data file in ADT format.
            instance_json_file_path (str): The file path to the JSON file containing instance data in ADT format
            category_mapping_file_path (Optional[str]): Path to the CSV file mapping category names to IDs, in the format of:
                {
                    $KEY_TO_MAP： [“cat_name”, category_id],
                    ...
                },
                where "KEY_TO_MAP" is one of strings of {"prototype_name", "category"}, set through conf.category_mapping_field_name
            camera_label_to_stream_ids (Dict[str, str]): A dictionary mapping camera labels to stream IDs.
            camera_label_to_pixel_transforms (Optional[Dict[str, Callable]]): A dictionary mapping camera labels to pixel transformation functions, if available.
            camera_label_to_calib (Optional[Dict[str, CameraCalibration]]): A dictionary mapping camera labels to camera calibration data, if available.

            conf (DictConfig): A configuration object containing settings for the processor, example yaml:
                ```
                category_mapping_field_name: "prototype_name" # {prototype_name (for ADT), category (for ASE)}
                bbox2d_num_samples_on_edge: 10                 # Number of samples on each edge of 2D bbox in distortion
                tolerance_ns: 1000000                          # Tolerance in nanoseconds for timestamp validation
                ```
        """
        self.conf = conf

        # Create a ADT data provider object, which will be used to load 3D BBOX GT data
        data_paths = AriaDigitalTwinDataPaths()
        data_paths.boundingboxes_2d_filepath = obb2_file_path
        data_paths.instances_filepath = instance_json_file_path

        self.adt_gt_provider = AriaDigitalTwinDataProvider(data_paths)

        self.camera_label_to_stream_ids = camera_label_to_stream_ids
        self.camera_label_to_pixel_transforms = camera_label_to_pixel_transforms
        self.camera_label_to_calibs = camera_label_to_calib

        self.category_mapping = (
            None
            if category_mapping_file_path is None
            else load_category_mapping_from_csv(category_mapping_file_path)
        )

    def _obtain_obj_category_info(self, instance_id: int) -> Tuple[str, int]:
        """
        Retrieves the category name and ID for a given instance ID.
        Args:
            instance_id (int): The instance ID for which to retrieve category information.
        Returns:
            Tuple[str, int]: A tuple containing the category name and category ID.
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
                category_id = int(self.category_mapping[key_to_map][1])
            else:
                category_name = "other"
                category_id = ATEK_OTHER_CATETORY_ID

        return category_name, category_id

    def _sample_points_on_bbox(
        self, bbox2d_range: np.array, num_points_on_edge: int
    ) -> torch.Tensor:
        """
        Samples points on the edges of a 2D bounding box.
        Args:
            bbox2d_range (np.array): An array containing the bounding box coordinates [xmin, xmax, ymin, ymax].
            num_points_on_edge (int): The number of points to sample along each edge of the bounding box.
        Returns:
            torch.Tensor: A tensor containing the sampled pixel coordinates.
        """
        # Create sampled pixel coordinates along each edge of the 2d bbox
        xmin, xmax, ymin, ymax = bbox2d_range
        x = torch.linspace(
            xmin, xmax, num_points_on_edge + 2
        )  # K+2 to include the corners
        y = torch.linspace(
            ymin, ymax, num_points_on_edge + 2
        )  # K+2 to include the corners

        # Create the points on the edges
        top_edge = torch.stack((x, torch.full_like(x, ymin)), dim=1)
        bottom_edge = torch.stack((x, torch.full_like(x, ymax)), dim=1)
        left_edge = torch.stack((torch.full_like(y, xmin), y), dim=1)
        right_edge = torch.stack((torch.full_like(y, xmax), y), dim=1)

        # Concatenate all the points
        pixel_coords = torch.cat((top_edge, bottom_edge, left_edge, right_edge), dim=0)

        return pixel_coords

    def _apply_transforms_to_bbox2d(
        self, camera_label: str, bbox2d_range: np.array
    ) -> torch.Tensor:
        """
        Apply the same transforms in AriaCameraProcessors to 2D bounding boxes.
        Returns a new 2dbbox that encloses the distorted box.
        """
        src_sampled_points = self._sample_points_on_bbox(
            bbox2d_range, self.conf.bbox2d_num_samples_on_edge
        )
        dst_sampled_points = self.camera_label_to_pixel_transforms[camera_label](
            src_sampled_points
        )

        # Get the new 2d bbox range
        dst_image_width, dst_image_height = self.camera_label_to_calibs[
            camera_label
        ].get_image_size()
        xmin, ymin = torch.min(dst_sampled_points, dim=0).values
        xmax, ymax = torch.max(dst_sampled_points, dim=0).values
        xmin = torch.clamp(xmin, 0, dst_image_width - 1)
        xmax = torch.clamp(xmax, 0, dst_image_width - 1)
        ymin = torch.clamp(ymin, 0, dst_image_height - 1)
        ymax = torch.clamp(ymax, 0, dst_image_height - 1)

        return torch.tensor([xmin, xmax, ymin, ymax], dtype=torch.float32)

    def get_gt_by_timestamp_ns(self, timestamp_ns: int) -> Optional[Dict]:
        """
        Retrieves the ground truth data for 2D bounding boxes by timestamp in nanoseconds.

        Args:
            timestamp_ns (int): The timestamp in nanoseconds for which to retrieve the ground truth data.

        Returns:
            Optional[Dict]: A dictionary containing the ground truth data for 2D bounding boxes if available and valid; otherwise, None.

            If not None, the returned dictionary will have the following structure:
            {
                "camera_label_1": {
                    "instance_ids": torch.Tensor (shape: [num_instances], int64)
                    "category_names": list[str],
                    "category_ids": torch.Tensor (shape: [num_instances], int64),
                    "visibility_ratios": torch.Tensor (shape: [num_instances], float32),
                    "box_ranges": torch.Tensor (shape: [num_instances, 4], float32, [xmin, xmax, ymin, ymax])
                },
                "camera_label_2": {
                    ...
                }
                ...
            }
            Each key in the outer dictionary corresponds to a camera label, while each inner dictionary contains instance information for that camera.

        Notes:
            Returns None if the data at the specified timestamp is not valid or does not meet the configured tolerances.
        """
        bbox2d_dict = {}
        for cam_label, stream_id in self.camera_label_to_stream_ids.items():

            bbox2d_with_dt = (
                self.adt_gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
                    timestamp_ns,
                    stream_id,
                )
            )
            # no valid data, skip current camera
            if (
                not bbox2d_with_dt.is_valid()
                or bbox2d_with_dt.dt_ns() > self.conf.tolerance_ns
            ):
                continue

            # initialize result dict for current camera
            num_visible_instances = len(bbox2d_with_dt.data())

            bbox2d_dict[cam_label] = {
                "instance_ids": torch.empty((num_visible_instances), dtype=torch.int64),
                "category_names": [],
                "category_ids": torch.empty((num_visible_instances), dtype=torch.int64),
                "visibility_ratios": torch.empty(
                    (num_visible_instances), dtype=torch.float32
                ),
                "box_ranges": torch.empty(
                    (num_visible_instances, 4), dtype=torch.float32
                ),
            }

            if num_visible_instances == 0:
                logger.debug(
                    f"No visible 2d bbox data for camera {cam_label} at {timestamp_ns}, skipping"
                )
                continue

            # pack 2d bbox data into the dict
            i_row = 0
            for instance_id, bbox2d_data in bbox2d_with_dt.data().items():
                # fill in instance id and category information
                cat_name, cat_id = self._obtain_obj_category_info(instance_id)
                bbox2d_dict[cam_label]["instance_ids"][i_row] = instance_id
                bbox2d_dict[cam_label]["category_names"].append(cat_name)
                bbox2d_dict[cam_label]["category_ids"][i_row] = cat_id
                bbox2d_dict[cam_label]["visibility_ratios"][
                    i_row
                ] = bbox2d_data.visibility_ratio
                bbox2d_dict[cam_label]["box_ranges"][i_row] = (
                    self._apply_transforms_to_bbox2d(cam_label, bbox2d_data.box_range)
                )
                i_row += 1
            assert (
                i_row == num_visible_instances
            ), f"camera {cam_label} filled number {i_row} != num of instances {num_visible_instances}, tensor contains initialized values, unsafe hence abort"

        # At least one camera should have valid data, or will return None
        valid_data_flag = False
        for per_cam_dict in bbox2d_dict.values():
            if len(per_cam_dict["category_names"]) > 0:
                valid_data_flag = True

        if not valid_data_flag:
            return None
        else:
            return bbox2d_dict

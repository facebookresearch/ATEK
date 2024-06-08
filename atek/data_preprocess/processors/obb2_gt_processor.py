# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from atek.data_preprocess.util.file_io_utils import load_category_mapping_from_csv

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
        Initialize the Obb2GtProcessor object.

        Args:
            obb2_file_path (str): The path to the OBB2 file, in ADT format: https://fburl.com/zh0p3egs
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
                category_id = int(self.category_mapping[key_to_map][1])
            else:
                category_name = "other"
                category_id = ATEK_OTHER_CATETORY_ID

        return category_name, category_id

    def _sample_points_on_bbox(
        self, bbox2d_range: np.array, num_points_on_edge: int
    ) -> torch.Tensor:
        """
        Sample points on the 2d bounding box, return pixel coords as tensor [N, 2]
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
        get obb2 GT by timestamp in nanoseconds
        """
        bbox2d_dict = {}
        for cam_label, stream_id in self.camera_label_to_stream_ids.items():
            bbox2d_dict[cam_label] = {}

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

            # pack 2d bbox data into a dict
            for instance_id, bbox2d_data in bbox2d_with_dt.data().items():
                single_bbox2d_dict = {}
                # fill in instance id and category information
                single_bbox2d_dict["instance_id"] = instance_id
                (
                    single_bbox2d_dict["category_name"],
                    single_bbox2d_dict["category_id"],
                ) = self._obtain_obj_category_info(instance_id)

                # Fill in 2d bbox information
                box_range = bbox2d_data.box_range  # [xmin, xmax, ymin, ymax]
                single_bbox2d_dict["visibility_ratio"] = bbox2d_data.visibility_ratio

                # 2d bbox needs to be undistorted -> rescaled -> rotated
                single_bbox2d_dict["box_range"] = self._apply_transforms_to_bbox2d(
                    cam_label, box_range
                )
                bbox2d_dict[cam_label][instance_id] = single_bbox2d_dict

        return bbox2d_dict

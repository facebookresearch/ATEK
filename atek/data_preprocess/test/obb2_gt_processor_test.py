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

import os
import unittest

import numpy as np

import torch
from atek.data_preprocess.processors.aria_camera_processor import AriaCameraProcessor

from atek.data_preprocess.processors.obb2_gt_processor import Obb2GtProcessor
from omegaconf import OmegaConf


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CATEGORY_MAPPING_PATH = os.getenv("CATEGORY_MAPPING_PATH")


class Obb2GtProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_obb2_gt_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        # First create an RGB camera processor, with all distortions turned on
        rgb_conf = OmegaConf.merge(
            conf.processors.rgb,
            {
                "undistort_to_linear_cam": True,
                "target_camera_resolution": [512, 512],
                "rotate_image_cw90deg": True,
            },
        )
        rgb_camera_processor = AriaCameraProcessor(
            video_vrs=os.path.join(TEST_DIR_PATH, "test_ADT_unit_test_sequence.vrs"),
            conf=rgb_conf,
        )
        rgb_calib = rgb_camera_processor.get_final_camera_calib()

        # Create the Obb2GtProcessor corresponding to the RGB camera
        obb2_gt_processor = Obb2GtProcessor(
            obb2_file_path=os.path.join(TEST_DIR_PATH, "test_2d_bounding_box.csv"),
            instance_json_file_path=os.path.join(TEST_DIR_PATH, "test_instances.json"),
            category_mapping_file_path=CATEGORY_MAPPING_PATH,
            camera_label_to_stream_ids={
                rgb_calib.get_label(): rgb_camera_processor.get_stream_id()
            },
            camera_label_to_pixel_transforms={
                rgb_calib.get_label(): rgb_camera_processor.get_pixel_transform()
            },
            camera_label_to_calib={rgb_calib.get_label(): rgb_calib},
            conf=conf.processors.obb_gt,
        )

        # Test for valid query
        queried_obb2_data = obb2_gt_processor.get_gt_by_timestamp_ns(
            timestamp_ns=87551170910700
        )
        self.assertEqual(len(queried_obb2_data), 1)  # 1 camera
        per_cam_dict = queried_obb2_data["camera-rgb"]
        gt_num_instances = 91  # 91 instances at this timestamp
        # Check tensor shapes
        self.assertEqual(len(per_cam_dict["category_names"]), gt_num_instances)
        self.assertEqual(
            per_cam_dict["instance_ids"].shape, torch.Size([gt_num_instances])
        )
        self.assertEqual(
            per_cam_dict["category_ids"].shape, torch.Size([gt_num_instances])
        )
        self.assertEqual(
            per_cam_dict["visibility_ratios"].shape, torch.Size([gt_num_instances])
        )
        self.assertEqual(
            per_cam_dict["box_ranges"].shape, torch.Size([gt_num_instances, 4])
        )

        # Check content for a specific instance
        instance_id = 6243788802362822
        ind = torch.where(per_cam_dict["instance_ids"] == instance_id)[0]
        self.assertTrue(len(ind) == 1)
        ind = ind.item()

        gt_visibility = torch.tensor([0.742435], dtype=torch.float32)
        self.assertEqual(per_cam_dict["category_names"][ind], "box")
        self.assertEqual(per_cam_dict["category_ids"][ind].item(), 31)
        self.assertTrue(
            torch.allclose(
                per_cam_dict["visibility_ratios"][ind], gt_visibility, atol=1e-5
            ),
        )
        # TODO: box_range cannot be easily checked due to rotation and distortion

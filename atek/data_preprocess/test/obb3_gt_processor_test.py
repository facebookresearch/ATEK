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

from atek.data_preprocess.processors.obb3_gt_processor import Obb3GtProcessor
from omegaconf import OmegaConf
from projectaria_tools.core.stream_id import StreamId


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CATEGORY_MAPPING_PATH = os.getenv("CATEGORY_MAPPING_PATH")


class Obb3GtProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_obb3_gt_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)
        conf.processors.obb_gt.category_mapping_field_name = "prototype_name"

        camera_label_to_stream_ids = {
            "camera-rgb": StreamId("214-1"),
            "camera-slam-left": StreamId("1201-1"),
        }

        obb3_gt_processor = Obb3GtProcessor(
            obb3_file_path=os.path.join(TEST_DIR_PATH, "test_3d_bounding_box.csv"),
            obb3_traj_file_path=os.path.join(
                TEST_DIR_PATH, "test_3d_bounding_box_traj.csv"
            ),
            obb2_file_path=os.path.join(TEST_DIR_PATH, "test_2d_bounding_box.csv"),
            instance_json_file_path=os.path.join(TEST_DIR_PATH, "test_instances.json"),
            category_mapping_file_path=CATEGORY_MAPPING_PATH,
            camera_label_to_stream_ids=camera_label_to_stream_ids,
            conf=conf.processors.obb_gt,
        )

        # Test for valid query
        queried_obb3_data = obb3_gt_processor.get_gt_by_timestamp_ns(
            timestamp_ns=87551170910700
        )
        rgb_visible_instances = queried_obb3_data["camera-rgb"]

        # Check bbox gt tensor shapes
        gt_num_instances = 91
        self.assertEqual(
            rgb_visible_instances["category_ids"].shape,
            torch.Size([gt_num_instances]),
        )
        self.assertEqual(len(rgb_visible_instances["category_names"]), gt_num_instances)
        self.assertEqual(
            rgb_visible_instances["object_dimensions"].shape,
            torch.Size([gt_num_instances, 3]),
        )
        self.assertEqual(
            rgb_visible_instances["ts_world_object"].shape,
            torch.Size([gt_num_instances, 3, 4]),
        )

        # Check the content of a specific instance 3d bbox
        instance_id = 4213328128795167
        ind = torch.where(rgb_visible_instances["instance_ids"] == instance_id)[0]
        self.assertTrue(len(ind) == 1)
        ind = ind.item()

        gt_cat_name = "other"
        gt_cat_id = 0
        gt_obj_dim = torch.tensor([0.35102427005, 0.17546123076, 0.24489420652])

        self.assertEqual(gt_cat_name, rgb_visible_instances["category_names"][ind])
        self.assertEqual(gt_cat_id, rgb_visible_instances["category_ids"][ind].item())
        self.assertTrue(
            torch.allclose(gt_obj_dim, rgb_visible_instances["object_dimensions"][ind])
        )

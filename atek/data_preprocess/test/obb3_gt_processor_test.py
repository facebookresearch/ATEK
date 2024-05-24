# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

import numpy as np

import torch

from atek.data_preprocess.processors.obb3_gt_processor import Obb3GtProcessor
from omegaconf import OmegaConf


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_PATH = os.getenv("CONFIG_PATH")
CATEGORY_MAPPING_PATH = os.getenv("CATEGORY_MAPPING_PATH")


class Obb3GtProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_obb3_gt_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)
        conf.processors.obb3_gt.category_mapping_field_name = "prototype_name"

        instance_id = 4709930779068635
        obb3_gt_processor = Obb3GtProcessor(
            obb3_file_path=os.path.join(TEST_DIR_PATH, "test_3d_bounding_box.csv"),
            obb3_traj_file_path=os.path.join(
                TEST_DIR_PATH, "test_3d_bounding_box_traj.csv"
            ),
            instance_json_file_path=os.path.join(TEST_DIR_PATH, "test_instances.json"),
            category_mapping_file_path=CATEGORY_MAPPING_PATH,
            conf=conf.processors.obb3_gt,
        )

        # Test for valid query
        queried_obb3_data = obb3_gt_processor.get_gt_by_timestamp_ns(
            timestamp_ns=87551170910700
        )
        self.assertEqual(len(queried_obb3_data), 349)  # 349 instances at this timestamp

        # Check the content of a specific instance 3d bbox
        gt_obj_dim = torch.tensor(
            [0.091384992 * 2, 1.00719845295 * 2, 0.39432799816 * 2]
        )
        gt_instance_dict = {
            "instance_id": instance_id,
            "category_name": "door",
            "category_id": 32,
            "object_dimensions": gt_obj_dim,
        }
        instance_dict = {
            key: queried_obb3_data[instance_id][key] for key in gt_instance_dict.keys()
        }

        self.assertEqual(instance_dict["instance_id"], instance_id)
        self.assertEqual(instance_dict["category_name"], "door")
        self.assertTrue(
            torch.allclose(instance_dict["object_dimensions"], gt_obj_dim, atol=1e-5)
        )
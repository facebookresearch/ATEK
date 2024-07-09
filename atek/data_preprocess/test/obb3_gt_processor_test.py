# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
        visible_instances = queried_obb3_data["instances_visible_to_cameras"]

        # Check bbox gt tensor shapes
        gt_num_instances = 349
        self.assertEqual(
            queried_obb3_data["obb3_all_category_ids"].shape,
            torch.Size([gt_num_instances]),
        )  # 349 instances at this timestamp
        self.assertEqual(
            len(queried_obb3_data["obb3_all_category_names"]), gt_num_instances
        )
        self.assertEqual(
            queried_obb3_data["obb3_all_object_dimensions"].shape,
            torch.Size([gt_num_instances, 3]),
        )
        self.assertEqual(
            queried_obb3_data["obb3_all_Ts_World_Object"].shape,
            torch.Size([gt_num_instances, 3, 4]),
        )

        # Check the content of a specific instance 3d bbox
        instance_id = 4709930779068635
        ind = torch.where(queried_obb3_data["obb3_all_instance_ids"] == instance_id)[0]
        self.assertTrue(len(ind) == 1)
        ind = ind.item()

        gt_cat_name = "door"
        gt_cat_id = 32
        gt_obj_dim = torch.tensor(
            [0.091384992 * 2, 1.00719845295 * 2, 0.39432799816 * 2]
        )

        self.assertEqual(gt_cat_name, queried_obb3_data["obb3_all_category_names"][ind])
        self.assertEqual(
            gt_cat_id, queried_obb3_data["obb3_all_category_ids"][ind].item()
        )
        self.assertTrue(
            torch.allclose(
                gt_obj_dim, queried_obb3_data["obb3_all_object_dimensions"][ind]
            )
        )

        # Check the content of visible instances
        self.assertEqual(visible_instances["camera-rgb"].shape, torch.Size([91]))
        self.assertEqual(visible_instances["camera-slam-left"].shape, torch.Size([69]))
        # Check that all visible instances are in instance list
        self.assertTrue(
            torch.isin(
                visible_instances["camera-rgb"],
                queried_obb3_data["obb3_all_instance_ids"],
            ).all()
        )
        self.assertTrue(
            torch.isin(
                visible_instances["camera-slam-left"],
                queried_obb3_data["obb3_all_instance_ids"],
            ).all()
        )

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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

        instance_id = 6243788802362822

        # Test for valid query
        queried_obb2_data = obb2_gt_processor.get_gt_by_timestamp_ns(
            timestamp_ns=87551170910700
        )
        self.assertEqual(len(queried_obb2_data), 1)  # 1 camera
        self.assertEqual(
            len(queried_obb2_data["camera-rgb"]), 91
        )  # 91 instances at this timestamp

        # Check the content of a specific instance 2 bbox
        gt_instance_dict = {
            "instance_id": instance_id,
            "category_name": "box",
            "category_id": 31,
            "visibility_ratio": 0.742435,
            "box_range": torch.tensor([920, 1066, 1157, 1308], dtype=torch.float32),
        }
        instance_dict = {
            key: queried_obb2_data["camera-rgb"][instance_id][key]
            for key in gt_instance_dict.keys()
        }

        self.assertEqual(instance_dict["instance_id"], gt_instance_dict["instance_id"])
        self.assertEqual(
            instance_dict["category_name"], gt_instance_dict["category_name"]
        )
        self.assertEqual(instance_dict["category_id"], gt_instance_dict["category_id"])
        self.assertTrue(
            np.allclose(
                instance_dict["visibility_ratio"],
                gt_instance_dict["visibility_ratio"],
                atol=1e-5,
            )
        )
        # check the tensor shape of the distorted box range
        self.assertEqual(torch.Size([4]), instance_dict["box_range"].shape)

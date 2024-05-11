# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

import numpy as np

import torch

from atek.data_preprocess.sample_builders.obb3_sample_builder import (
    ObjectDetection3dSampleBuilder,
)
from omegaconf import OmegaConf


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_PATH = os.getenv("CONFIG_PATH")


class Obb3SampleBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_obb3_sample(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        sample_builder = ObjectDetection3dSampleBuilder(
            conf=conf.processors,
            vrs_file=os.path.join(TEST_DIR_PATH, "test_ADT_unit_test_sequence.vrs"),
            mps_files={
                "mps_closedloop_traj_file": os.path.join(
                    TEST_DIR_PATH, "test_ADT_trajectory.csv"
                ),
            },
            gt_files={
                "obb3_file": os.path.join(TEST_DIR_PATH, "test_3d_bounding_box.csv"),
                "obb3_traj_file": os.path.join(
                    TEST_DIR_PATH, "test_3d_bounding_box_traj.csv"
                ),
                "instance_json_file": os.path.join(
                    TEST_DIR_PATH, "test_instances.json"
                ),
            },
        )

        queried_sample = sample_builder.get_sample_by_timestamp_ns(
            timestamp_ns=87551170910000
        )

        self.assertTrue(queried_sample is not None)
        self.assertTrue(queried_sample.camera_rgb is not None)
        self.assertTrue(queried_sample.camera_et_left is None)  # No ET data
        self.assertTrue(queried_sample.mps_traj_data is not None)
        self.assertTrue(queried_sample.gt_data is not None)

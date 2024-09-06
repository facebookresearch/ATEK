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

from atek.data_preprocess.sample_builders.obb_sample_builder import ObbSampleBuilder

from omegaconf import OmegaConf


# test data paths
TEST_DIR_PATH = os.path.join(os.getenv("TEST_FOLDER"))
CONFIG_DIR = os.getenv("CONFIG_FOLDER")


class ObbSampleBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_obb3_sample(self) -> None:
        conf = OmegaConf.load(os.path.join(CONFIG_DIR, "obb_preprocess_base.yaml"))

        sample_builder = ObbSampleBuilder(
            conf=conf.processors,
            vrs_file=os.path.join(TEST_DIR_PATH, "test_ADT_unit_test_sequence.vrs"),
            sequence_name="test",
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
                "obb2_file": os.path.join(TEST_DIR_PATH, "test_2d_bounding_box.csv"),
                "instance_json_file": os.path.join(
                    TEST_DIR_PATH, "test_instances.json"
                ),
            },
            # TODO: add depth testing
        )

        queried_sample = sample_builder.get_sample_by_timestamp_ns(
            timestamp_ns=87551170910000
        )

        self.assertTrue(queried_sample is not None)
        self.assertEqual(queried_sample.sequence_name, "test")
        self.assertTrue(queried_sample.camera_rgb is not None)
        self.assertFalse(hasattr(queried_sample, "camera_et_left"))  # No ET data
        self.assertTrue(queried_sample.mps_traj_data is not None)
        self.assertCountEqual(queried_sample.gt_data.keys(), ["obb2_gt", "obb3_gt"])

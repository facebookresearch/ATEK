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

import torch

from atek.data_preprocess.processors.aria_camera_processor import AriaCameraProcessor
from omegaconf import OmegaConf


# test data paths
TEST_VRS_PATH = os.path.join(
    os.getenv("TEST_FOLDER"), "test_ADT_unit_test_sequence.vrs"
)
CONFIG_PATH = os.getenv("CONFIG_PATH")


class AriaCameraProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def _single_case_test_get_image_data(
        self, camera_processor, gt_timestamps_ns, gt_frame_id, gt_image_shape
    ):
        """
        A helper function to perform a test on a single camera processor
        """
        # Expect to success, only 20ns away
        query_timestamps = gt_timestamps_ns + 20
        maybe_result = camera_processor.get_image_data_by_timestamps_ns(
            timestamps_ns=query_timestamps.tolist()
        )
        self.assertTrue(maybe_result is not None)

        self.assertTrue(torch.allclose(maybe_result.frame_ids, gt_frame_id, atol=1))
        self.assertTrue(
            torch.allclose(maybe_result.capture_timestamps_ns, gt_timestamps_ns, atol=1)
        )
        self.assertEqual(maybe_result.images.shape, gt_image_shape)

    def test_get_image_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        # Test for Slam left
        slam_camera_processor = AriaCameraProcessor(
            TEST_VRS_PATH, conf.processors.slam_left
        )
        self._single_case_test_get_image_data(
            slam_camera_processor,
            gt_timestamps_ns=torch.tensor(
                [87551270894700, 87551337550700], dtype=torch.int64
            ),
            gt_frame_id=torch.tensor([1335, 1337], dtype=torch.int64),
            gt_image_shape=torch.Size([2, 1, 480, 640]),
        )

        # Test for rotated SLAM left
        slam_conf_2 = OmegaConf.merge(
            conf.processors.slam_left, {"rotate_image_cw90deg": True}
        )
        slam_camera_processor_2 = AriaCameraProcessor(TEST_VRS_PATH, slam_conf_2)
        self._single_case_test_get_image_data(
            slam_camera_processor_2,
            gt_timestamps_ns=torch.tensor([87551337550700], dtype=torch.int64),
            gt_frame_id=torch.tensor([1337], dtype=torch.int64),
            gt_image_shape=torch.Size([1, 1, 640, 480]),
        )

        # Test for RGB camera
        rgb_conf = OmegaConf.merge(
            conf.processors.rgb, {"target_camera_resolution": [512, 512]}
        )
        rgb_camera_processor = AriaCameraProcessor(TEST_VRS_PATH, rgb_conf)
        self._single_case_test_get_image_data(
            rgb_camera_processor,
            gt_timestamps_ns=torch.tensor([87551337447475], dtype=torch.int64),
            gt_frame_id=torch.tensor([1335], dtype=torch.int64),
            gt_image_shape=torch.Size([1, 3, 512, 512]),
        )

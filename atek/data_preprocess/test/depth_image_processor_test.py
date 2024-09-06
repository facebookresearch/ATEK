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
from atek.data_preprocess.processors.depth_image_processor import DepthImageProcessor
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode


# test data paths
TEST_FOLDER = os.getenv("TEST_FOLDER")
CONFIG_PATH = os.getenv("CONFIG_PATH")


class DepthImageProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def _single_case_test_get_depth_data(
        self, camera_processor, gt_timestamps_ns, gt_frame_id, gt_image_shape
    ):
        """
        A helper function to perform a test on a single camera processor
        """
        # Expect to success, only 20ns away
        query_timestamps = gt_timestamps_ns + 20
        maybe_result = camera_processor.get_depth_data_by_timestamps_ns(
            timestamps_ns=query_timestamps.tolist()
        )
        self.assertTrue(maybe_result is not None)

        self.assertTrue(torch.allclose(maybe_result.frame_ids, gt_frame_id, atol=1))
        self.assertTrue(
            torch.allclose(maybe_result.capture_timestamps_ns, gt_timestamps_ns, atol=1)
        )
        self.assertEqual(maybe_result.images.shape, gt_image_shape)

    def test_get_depth_image_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        # Obtain image transformations from AriaCameraProcessor
        rgb_conf = conf.processors.rgb
        rgb_conf = OmegaConf.merge(rgb_conf, {"undistort_to_linear_camera": True})
        rgb_conf = OmegaConf.merge(rgb_conf, {"target_camera_resolution": [704, 704]})
        rgb_conf = OmegaConf.merge(rgb_conf, {"rotate_image_cw90deg": True})
        rgb_camera_processor = AriaCameraProcessor(
            video_vrs=os.path.join(TEST_FOLDER, "test_ADT_unit_test_sequence.vrs"),
            conf=rgb_conf,
        )
        depth_image_transform = rgb_camera_processor.get_image_transform(
            rescale_interpolation=InterpolationMode.NEAREST
        )

        # Construct DepthImageProcessor
        depth_conf = conf.processors.rgb_depth
        # replace type id with exact stream id, and turn on depth conversion, for ADT data.
        depth_conf.pop("depth_stream_type_id")
        depth_conf = OmegaConf.merge(
            depth_conf, {"depth_stream_id": "345-1", "convert_zdepth_to_distance": True}
        )
        rgb_depth_processor = DepthImageProcessor(
            depth_vrs=os.path.join(TEST_FOLDER, "test_ADT_depth_rgb_only.vrs"),
            image_transform=depth_image_transform,
            depth_camera_label="camera-rgb-depth",
            depth_camera_calib=rgb_camera_processor.get_final_camera_calib(),
            conf=depth_conf,
        )

        self._single_case_test_get_depth_data(
            rgb_depth_processor,
            gt_timestamps_ns=torch.tensor(
                [87551204237999, 87551237566000], dtype=torch.int64
            ),
            gt_frame_id=torch.tensor([1, 2], dtype=torch.int64),
            gt_image_shape=torch.Size([2, 1, 704, 704]),
        )

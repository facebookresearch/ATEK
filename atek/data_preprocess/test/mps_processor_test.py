# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

import torch

from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from omegaconf import OmegaConf

# test data paths
TEST_TRAJ_FILE_PATH = os.path.join(os.getenv("TEST_FOLDER"), "test_mps_traj.csv")
CONFIG_PATH = os.getenv("CONFIG_PATH")


class MpsTrajProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_traj_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        mps_traj_processor = MpsTrajProcessor(
            mps_closedloop_traj_file=TEST_TRAJ_FILE_PATH,
            conf=conf.processors.mps_traj,
        )

        # Test for query for closed loop traj data
        query_timestamp = 100_001_000  # actual timestamp is 100_000_000
        T_world_device, capture_timestamp, gravity_in_world = (
            mps_traj_processor.get_closed_loop_pose_by_timestamp_ns(query_timestamp)
        )
        gt_translation = torch.tensor(
            [-7.7852104154284, 0.536540244562757, 1.2436284253971863],
            dtype=torch.float32,
        )
        gt_gravity = torch.tensor([0, 0, -9.81], dtype=torch.float32)

        self.assertTrue(
            torch.allclose(T_world_device[:, :, 3], gt_translation, atol=1e-6)
        )
        self.assertEqual(capture_timestamp.item(), 100_000_000)
        self.assertTrue(torch.allclose(gravity_in_world, gt_gravity))

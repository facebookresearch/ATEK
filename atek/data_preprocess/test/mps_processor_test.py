# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

import torch
from atek.data_preprocess.processors.mps_semidense_processor import (
    MpsSemiDenseProcessor,
)

from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from omegaconf import OmegaConf

# test data paths
TEST_DIR = os.getenv("TEST_FOLDER")
CONFIG_PATH = os.getenv("CONFIG_PATH")


class MpsTrajProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_traj_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)

        mps_traj_processor = MpsTrajProcessor(
            mps_closedloop_traj_file=os.path.join(TEST_DIR, "test_mps_traj.csv"),
            conf=conf.processors.mps_traj,
        )

        # Test for query for closed loop traj data
        query_timestamp = 100_001_000  # actual timestamp is 100_000_000
        Ts_world_device, capture_timestamps, gravity_in_world = (
            mps_traj_processor.get_closed_loop_pose_by_timestamps_ns([query_timestamp])
        )
        gt_translation = torch.tensor(
            [-7.7852104154284, 0.536540244562757, 1.2436284253971863],
            dtype=torch.float32,
        )
        gt_gravity = torch.tensor([0, 0, -9.81], dtype=torch.float32)

        self.assertTrue(
            torch.allclose(
                Ts_world_device[0, :, 3].squeeze(), gt_translation, atol=1e-6
            )
        )
        self.assertTrue(
            torch.allclose(
                capture_timestamps,
                torch.tensor([100_000_000], dtype=torch.int64),
                atol=10,
            )
        )

        self.assertTrue(torch.allclose(gravity_in_world, gt_gravity))


class MpsSemiDenseProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_semidense_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)
        conf = conf.processors.mps_semidense
        OmegaConf.update(conf, "selected", True)
        OmegaConf.update(conf, "tolerance_ns", 10_000_000)

        mps_semidense_processor = MpsSemiDenseProcessor(
            mps_semidense_points_file=os.path.join(
                TEST_DIR, "test_mps_semidense_points.csv"
            ),
            mps_semidense_observations_file=os.path.join(
                TEST_DIR, "test_mps_semidense_observations.csv"
            ),
            conf=conf,
        )

        # test case 1: query exact timestamps
        maybe_result = mps_semidense_processor.get_semidense_points_by_timestamps_ns(
            timestamps_ns=[1_000_000_000, 2_000_000_000]
        )
        self.assertTrue(maybe_result is not None)
        points, points_inv_dist = maybe_result

        # check tensor sizes
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].shape, torch.Size([2, 3]))  # 2 points in frame 0
        self.assertEqual(points[1].shape, torch.Size([3, 3]))  # 3 points in frame 1
        points_0_gt = torch.tensor(
            [[1.1, 1.2, 1.3], [37.1, 37.2, 37.3]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(points[0], points_0_gt, atol=1e-6))

        self.assertEqual(len(points_inv_dist), 2)
        self.assertEqual(
            points_inv_dist[0].shape, torch.Size([2])
        )  # 2 points in frame 0
        self.assertEqual(
            points_inv_dist[1].shape, torch.Size([3])
        )  # 3 points in frame 1
        points_inv_dist_0_gt = torch.tensor([0.003901, 0.00644], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(points_inv_dist[0], points_inv_dist_0_gt, atol=1e-6)
        )

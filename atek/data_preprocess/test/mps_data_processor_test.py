# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
import unittest

import torch

from atek.data_preprocess.mps_data_processor import MpsDataProcessor


# test data paths
TEST_TRAJ_FILE_PATH = os.path.join(os.getenv("TEST_FOLDER"), "test_mps_traj.csv")
TEST_SEMIDENSE_POINT_FILE_PATH = os.path.join(
    os.getenv("TEST_FOLDER"), "test_mps_semidense_points.csv"
)
TEST_SEMIDENSE_OBSERVATION_FILE_PATH = os.path.join(
    os.getenv("TEST_FOLDER"), "test_mps_semidense_observations.csv"
)


class MpsDataProcessorTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_get_nearest_semidense_points(self) -> None:
        mps_data_processor = MpsDataProcessor(
            name="test_mps",
            trajectory_file=TEST_TRAJ_FILE_PATH,
            semidense_global_point_file=TEST_SEMIDENSE_POINT_FILE_PATH,
            semidense_observation_file=TEST_SEMIDENSE_OBSERVATION_FILE_PATH,
        )

        # test case 1: query exact timestamps
        points_df = mps_data_processor.get_nearest_semidense_points(
            timestamps_ns=[1_000_000_000, 2_000_000_000]
        )
        points = points_df["points_world"].to_list()
        # check tensor sizes
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].shape, torch.Size([2, 3]))  # 2 points in frame 0
        self.assertEqual(points[1].shape, torch.Size([3, 3]))  # 3 points in frame 1
        points_0_gt = torch.tensor(
            [[1.1, 1.2, 1.3], [37.1, 37.2, 37.3]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(points[0], points_0_gt, atol=1e-6))

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
from atek.data_preprocess.processors.mps_online_calib_processor import (
    MpsOnlineCalibProcessor,
)
from atek.data_preprocess.processors.mps_semidense_processor import (
    MpsSemiDenseProcessor,
)

from atek.data_preprocess.processors.mps_traj_processor import MpsTrajProcessor
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

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
        points = maybe_result.points_world
        points_dist = maybe_result.points_dist_std
        points_inv_dist = maybe_result.points_inv_dist_std

        # check tensor sizes
        self.assertEqual(len(points), 2)
        self.assertEqual(points[0].shape, torch.Size([2, 3]))  # 2 points in frame 0
        self.assertEqual(points[1].shape, torch.Size([3, 3]))  # 3 points in frame 1
        points_0_gt = torch.tensor(
            [[1.1, 1.2, 1.3], [37.1, 37.2, 37.3]], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(points[0], points_0_gt, atol=1e-6))

        # check dist std
        self.assertEqual(len(points_dist), 2)
        self.assertEqual(points_dist[0].shape, torch.Size([2]))  # 2 points in frame 0
        self.assertEqual(points_dist[1].shape, torch.Size([3]))  # 3 points in frame 1
        points_dist_0_gt = torch.tensor([0.002598, 0.035042], dtype=torch.float32)
        self.assertTrue(torch.allclose(points_dist[0], points_dist_0_gt, atol=1e-6))
        points_dist_1_gt = torch.tensor(
            [0.017871, 0.002598, 0.013965], dtype=torch.float32
        )
        self.assertTrue(torch.allclose(points_dist[1], points_dist_1_gt, atol=1e-6))

        # check inv dist std
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
        points_inv_dist_1_gt = torch.tensor(
            [0.001829, 0.003901, 0.004357], dtype=torch.float32
        )
        self.assertTrue(
            torch.allclose(points_inv_dist[1], points_inv_dist_1_gt, atol=1e-6)
        )

        # check tracking timestamps
        gt_timestamps = torch.tensor([1_000_000_000, 2_000_000_000], dtype=torch.int64)
        self.assertTrue(
            torch.allclose(maybe_result.capture_timestamps_ns, gt_timestamps)
        )

        # check vol max and min
        self.assertTrue(isinstance(maybe_result.points_volumn_min, torch.Tensor))
        self.assertEqual(maybe_result.points_volumn_min.shape, torch.Size([3]))
        self.assertTrue(isinstance(maybe_result.points_volumn_max, torch.Tensor))
        self.assertEqual(maybe_result.points_volumn_max.shape, torch.Size([3]))


class MpsOnlineCalibProcessorTest(unittest.TestCase):
    """
    Tests for reading online calibration data from jsonl files, we need to test the MpsOnlineCalibData:
    capture_timestamps_ns=capture_timestamp_tensor,
    utc_timestamps_ns=utc_timestamp_tensor, we don't need to test this, since in the test file, this field is not valid
    projection_params=projection_params_tensor,
    ts_device_camera=ts_device_camera_tensor,
    """

    def setUp(self) -> None:
        super().setUp()

    def test_get_online_calib_data(self) -> None:
        conf = OmegaConf.load(CONFIG_PATH)
        mps_online_calib_processor = MpsOnlineCalibProcessor(
            mps_online_calib_data_file_path=os.path.join(
                TEST_DIR, "test_mps_online_calibration.jsonl"
            ),
            conf=conf.processors.mps_online_calib,
        )

        mps_online_calib_data = (
            mps_online_calib_processor.get_online_calibration_by_timestamps_ns(
                timestamps_ns=[14580434854000]
            )
        )
        self.assertTrue(mps_online_calib_data is not None)
        self.assertEqual(
            mps_online_calib_data.projection_params.shape, torch.Size([1, 3, 15])
        )
        self.assertEqual(
            mps_online_calib_data.ts_device_camera.shape, torch.Size([1, 3, 3, 4])
        )
        self.assertEqual(
            mps_online_calib_data.capture_timestamps_ns,
            torch.tensor([14580434854000], dtype=torch.int64),
        )
        self._test_translation_and_quaternion(mps_online_calib_data.ts_device_camera)
        self._test_projection_params(mps_online_calib_data.projection_params)
        return

    def _test_translation_and_quaternion(self, ts_device_camera_tensor):
        expected_translations = [
            [0.0004328977150926845, -0.00009016214024268332, -0.00004893689922337574],
            [0.006082919423532053, -0.11158645596563903, -0.0878174381882055],
            [-0.004280716004834114, -0.01184173316010912, -0.005113984064522773],
        ]

        expected_quaternions = [
            [
                0.9999746350011375,
                0.006457900925989656,
                -0.002989648005019753,
                -0.0002947452769143969,
            ],
            [
                0.7875179103510879,
                0.6156645328476295,
                0.006744113116638467,
                0.026967402693551194,
            ],
            [
                0.94426112424187,
                0.3241397857173833,
                0.0402122625965168,
                0.04107678781790648,
            ],
        ]
        for i in range(len(expected_translations)):
            expected_translation = np.array(expected_translations[i])
            expected_quaternion = np.array(expected_quaternions[i])
            # Extract the translation and rotation matrix from the tensor
            matrix = ts_device_camera_tensor[0, i, :, :].numpy()
            actual_translation = matrix[:3, 3]
            rotation_matrix = matrix[:3, :3]
            # Convert rotation matrix to quaternion
            actual_quaternion = R.from_matrix(rotation_matrix).as_quat()
            # Convert to the same format as the expected quaternion (scalar-first)
            actual_quaternion = np.roll(actual_quaternion, shift=1)
            np.testing.assert_allclose(
                actual_translation,
                expected_translation,
                atol=1e-5,
                err_msg=f"Translation mismatch at index {i}",
            )
            # Compare quaternions
            np.testing.assert_allclose(
                actual_quaternion,
                expected_quaternion,
                atol=1e-5,
                err_msg=f"Quaternion mismatch at index {i}",
            )

    def _test_projection_params(self, projection_params):
        projection_params_expect_tesnor = torch.tensor(
            [
                [
                    [
                        241.38275146484375,
                        316.2264709472656,
                        236.6782989501953,
                        -0.02765783481299877,
                        0.10371126979589462,
                        -0.07342791557312012,
                        0.012858348898589611,
                        0.0013407421065494418,
                        -0.000457167363492772,
                        0.0004445339145604521,
                        -0.0019325445173308253,
                        0.00015349453315138817,
                        0.00026054191403090954,
                        0.0033760429359972477,
                        0.00013414736895356327,
                    ],
                    [
                        241.03900146484375,
                        317.6932373046875,
                        238.18008422851562,
                        -0.02463776431977749,
                        0.09580568224191666,
                        -0.06505261361598969,
                        0.009101307950913906,
                        0.0019199964590370655,
                        -0.00044897140469402075,
                        0.0015517562860623002,
                        -0.0005500924889929593,
                        -0.000873647746630013,
                        0.00021656886383425444,
                        0.0011815830366685987,
                        0.0001968961296370253,
                    ],
                    [
                        1221.882015735315,
                        1462.729668220901,
                        1465.929424505965,
                        0.4060356696288849,
                        -0.489948419647729,
                        0.1745652818132035,
                        1.132983686620576,
                        -1.701635218233742,
                        0.6511555293441647,
                        0.0006211469747214578,
                        0.00001932200015697112,
                        -0.00001485525650871087,
                        0.0002601225712292815,
                        -0.0006582109778598294,
                        0.00003761395141407565,
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertTrue(
            torch.allclose(
                projection_params,
                projection_params_expect_tesnor,
                atol=1e-6,
            )
        )

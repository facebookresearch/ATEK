# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsSemiDensePointData,
    MpsTrajData,
    MultiFrameCameraData,
)


class MultiFrameCameraDataTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flatten_to_dict(self) -> None:
        data = MultiFrameCameraData()
        data.images = torch.randn(4, 1, 10, 10)
        data.frame_ids = torch.tensor([1, 2, 3, 4])

        data.camera_label = "test-camera"
        data.projection_params = torch.randn(20, 1)

        data_dict = data.to_flatten_dict()
        expected_keys = [
            "mfcd#test-camera+images_0.jpeg",
            "mfcd#test-camera+images_1.jpeg",
            "mfcd#test-camera+images_2.jpeg",
            "mfcd#test-camera+images_3.jpeg",
            "mfcd#test-camera+frame_ids.pth",
            "mfcd#test-camera+camera_label.txt",
            "mfcd#test-camera+projection_params.pth",
        ]
        self.assertCountEqual(data_dict.keys(), expected_keys)


class TestMpsSemiDensePointData(unittest.TestCase):
    def test_to_flatten_dict(self):
        # Create an instance of MpsSemiDensePointData with sample data
        point_data = MpsSemiDensePointData(
            points_world=[
                torch.tensor([[1, 2, 3], [4, 5, 6]]),
                torch.tensor([[7, 8, 9]]),
            ],
            points_inv_dist_std=[torch.tensor([0.1, 0.2]), torch.tensor([0.3])],
            points_dist_std=[torch.tensor([0.5, 0.4]), torch.tensor([0.3])],
            capture_timestamps_ns=torch.tensor([1, 2]),
        )

        # Call the method
        flatten_dict = point_data.to_flatten_dict()

        # Check the keys in the flatten_dict
        expected_keys = [
            "msdpd#points_world_lengths.pth",
            "msdpd#stacked_points_world.pth",
            "msdpd#stacked_points_inv_dist_std.pth",
            "msdpd#stacked_points_dist_std.pth",
            "msdpd#capture_timestamps_ns.pth",
        ]
        self.assertCountEqual(flatten_dict.keys(), expected_keys)

        # Check value
        self.assertTrue(
            torch.allclose(
                flatten_dict["msdpd#points_world_lengths.pth"],
                torch.tensor([2, 1], dtype=torch.int64),
                atol=0,
            )
        )

        # Check shape
        self.assertEqual(
            flatten_dict["msdpd#stacked_points_world.pth"].shape, torch.Size([3, 3])
        )
        self.assertEqual(
            flatten_dict["msdpd#stacked_points_inv_dist_std.pth"].shape, torch.Size([3])
        )


class AtekDataSampleTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    def test_flatten_to_dict(self) -> None:
        rgb_data = MultiFrameCameraData(
            images=torch.randn(4, 3, 10, 10),
            camera_label="camera-rgb",
            projection_params=torch.randn(20, 1),
        )

        slam_left_data = MultiFrameCameraData(
            images=torch.randn(4, 1, 10, 12),
            camera_label="camera-slam-left",
            frame_ids=torch.tensor([1, 2, 3, 4]),
        )

        mps_traj_data = MpsTrajData(
            Ts_World_Device=torch.randn(1, 3, 4),
            gravity_in_world=torch.tensor([0, 0, -9.81]),
        )

        gt_data = {"key_1": "value_1", "key_2": "value_2"}

        data = AtekDataSample(
            camera_rgb=rgb_data,
            camera_slam_left=slam_left_data,
            mps_traj_data=mps_traj_data,
            gt_data=gt_data,
        )

        data_dict = data.to_flatten_dict()
        expected_keys = [
            "mfcd#camera-rgb+images_0.jpeg",
            "mfcd#camera-rgb+images_1.jpeg",
            "mfcd#camera-rgb+images_2.jpeg",
            "mfcd#camera-rgb+images_3.jpeg",
            "mfcd#camera-rgb+camera_label.txt",
            "mfcd#camera-rgb+projection_params.pth",
            "mfcd#camera-slam-left+images_0.jpeg",
            "mfcd#camera-slam-left+images_1.jpeg",
            "mfcd#camera-slam-left+images_2.jpeg",
            "mfcd#camera-slam-left+images_3.jpeg",
            "mfcd#camera-slam-left+camera_label.txt",
            "mfcd#camera-slam-left+frame_ids.pth",
            "mtd#ts_world_device.pth",
            "mtd#gravity_in_world.pth",
            "gtdata.json",
        ]
        self.assertCountEqual(data_dict.keys(), expected_keys)

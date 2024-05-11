# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
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
            "MFCD#test-camera+images_0.jpeg",
            "MFCD#test-camera+images_1.jpeg",
            "MFCD#test-camera+images_2.jpeg",
            "MFCD#test-camera+images_3.jpeg",
            "MFCD#test-camera+frame_ids.pth",
            "MFCD#test-camera+camera_label.txt",
            "MFCD#test-camera+projection_params.pth",
        ]
        self.assertCountEqual(data_dict.keys(), expected_keys)


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
            "MFCD#camera-rgb+images_0.jpeg",
            "MFCD#camera-rgb+images_1.jpeg",
            "MFCD#camera-rgb+images_2.jpeg",
            "MFCD#camera-rgb+images_3.jpeg",
            "MFCD#camera-rgb+camera_label.txt",
            "MFCD#camera-rgb+projection_params.pth",
            "MFCD#camera-slam-left+images_0.jpeg",
            "MFCD#camera-slam-left+images_1.jpeg",
            "MFCD#camera-slam-left+images_2.jpeg",
            "MFCD#camera-slam-left+images_3.jpeg",
            "MFCD#camera-slam-left+camera_label.txt",
            "MFCD#camera-slam-left+frame_ids.pth",
            "MTD#Ts_World_Device.pth",
            "MTD#gravity_in_world.pth",
            "GtData.json",
        ]
        self.assertCountEqual(data_dict.keys(), expected_keys)

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.


import unittest

import torch

from atek.data_preprocess.atek_data_sample import (
    AtekDataSample,
    MpsOnlineCalibData,
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

        mps_online_calib_data = MpsOnlineCalibData(
            capture_timestamps_ns=torch.tensor([14580434854000]),
            utc_timestamps_ns=torch.tensor([0], dtype=torch.int64),
            projection_params=torch.tensor(
                [
                    [
                        [
                            2.4138e02,
                            3.1623e02,
                            2.3668e02,
                            -2.7658e-02,
                            1.0371e-01,
                            -7.3428e-02,
                            1.2858e-02,
                            1.3407e-03,
                            -4.5717e-04,
                            4.4453e-04,
                            -1.9325e-03,
                            1.5349e-04,
                            2.6054e-04,
                            3.3760e-03,
                            1.3415e-04,
                        ],
                        [
                            2.4104e02,
                            3.1769e02,
                            2.3818e02,
                            -2.4638e-02,
                            9.5806e-02,
                            -6.5053e-02,
                            9.1013e-03,
                            1.9200e-03,
                            -4.4897e-04,
                            1.5518e-03,
                            -5.5009e-04,
                            -8.7365e-04,
                            2.1657e-04,
                            1.1816e-03,
                            1.9690e-04,
                        ],
                        [
                            1.2219e03,
                            1.4627e03,
                            1.4659e03,
                            4.0604e-01,
                            -4.8995e-01,
                            1.7457e-01,
                            1.1330e00,
                            -1.7016e00,
                            6.5116e-01,
                            6.2115e-04,
                            1.9322e-05,
                            -1.4855e-05,
                            2.6012e-04,
                            -6.5821e-04,
                            3.7614e-05,
                        ],
                    ]
                ],
                dtype=torch.float64,
            ),
            t_device_camera=torch.tensor(
                [
                    [
                        [
                            [9.9998e-01, 5.5086e-04, -5.9830e-03, 4.3290e-04],
                            [-6.2809e-04, 9.9992e-01, -1.2914e-02, -9.0162e-05],
                            [5.9753e-03, 1.2917e-02, 9.9990e-01, -4.8937e-05],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ],
                        [
                            [9.9845e-01, -3.4170e-02, 4.3828e-02, 6.0829e-03],
                            [5.0779e-02, 2.4046e-01, -9.6933e-01, -1.1159e-01],
                            [2.2584e-02, 9.7006e-01, 2.4182e-01, -8.7817e-02],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ],
                        [
                            [9.9339e-01, -5.1506e-02, 1.0257e-01, -4.2807e-03],
                            [1.0364e-01, 7.8649e-01, -6.0884e-01, -1.1842e-02],
                            [-4.9313e-02, 6.1545e-01, 7.8663e-01, -5.1140e-03],
                            [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
                        ],
                    ]
                ]
            ),
        )

        gt_data = {"key_1": "value_1", "key_2": "value_2"}

        data = AtekDataSample(
            camera_rgb=rgb_data,
            camera_slam_left=slam_left_data,
            mps_traj_data=mps_traj_data,
            gt_data=gt_data,
            mps_online_calib_data=mps_online_calib_data,
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
            "moc#capture_timestamps_ns.pth",
            "moc#utc_timestamps_ns.pth",
            "moc#projection_params.pth",
            "moc#t_device_camera.pth",
        ]
        self.assertCountEqual(data_dict.keys(), expected_keys)

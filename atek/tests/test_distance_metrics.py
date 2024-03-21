# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
import unittest

import torch

from atek.evaluation.math_metrics.distance_metrics import (
    chamfer_distance,
    chamfer_distance_single,
    hungarian_distance,
    hungarian_distance_single,
)
from atek.tests.test_utils import get_random_dim_and_T
from atek.utils.obb3 import Obb3
from pytorch3d.transforms import euler_angles_to_matrix


class TestDistanceMetrics(unittest.TestCase):
    async def helper(
        self,
        obb3_a: Obb3,
        obb3_b: Obb3,
        gt_dist_chamfer: float,
        gt_dist_hungarian: float,
        i: int = 0,
        j: int = 0,
    ) -> None:
        dist_chamfer = chamfer_distance_single(
            obb3_a.bb3_in_ref_frame[i], obb3_b.bb3_in_ref_frame[i]
        )
        dist_hungarian = hungarian_distance_single(
            obb3_a.bb3_in_ref_frame[j], obb3_b.bb3_in_ref_frame[j]
        )

        self.assertAlmostEqual(dist_chamfer, gt_dist_chamfer, delta=1e-6)
        self.assertAlmostEqual(dist_hungarian, gt_dist_hungarian, delta=1e-6)

    async def test_distance_single(self) -> None:
        size_a = [[1, 1, 1]]
        t_ref_a = [[0, 0, 0]]
        R_ref_a = euler_angles_to_matrix([[0, 0, 0]], convention="XYZ")
        instance_id_a = [1]
        category_a = [1]
        obb3_a = Obb3(size_a, t_ref_a, R_ref_a, instance_id_a, category_a)

        """ ========================================================================
        test case - same box
        ======================================================================== """
        self.helper(obb3_a, obb3_a, 0.0, 0.0)

        """ ========================================================================
        test case - scale difference
        ======================================================================== """
        size_b = [[2, 2, 2]]
        t_ref_b = [[0, 0, 0]]
        R_ref_b = euler_angles_to_matrix([[0, 0, 0]], convention="XYZ")
        instance_id_b = [2]
        category_b = [1]
        obb3_b = Obb3(size_b, t_ref_b, R_ref_b, instance_id_b, category_b)

        self.helper(obb3_a, obb3_b, 3.0, 1.5)

        """ ========================================================================
        test case - translation in XYZ axis (having overlap)
        ======================================================================== """
        obb3_b.size = torch.tensor([[1, 1, 1]])
        obb3_b.t_ref_obj = torch.tensor([[0.3, 0.4, 0.5]])

        self.helper(obb3_a, obb3_b, 2.4, 1.2)

        """ ========================================================================
        test case - translation in XYZ axes (no overlap)
        ======================================================================== """
        obb3_b.t_ref_obj = torch.tensor([[1, 2, 3]])

        self.helper(obb3_a, obb3_b, 9.0, 6.0)

        """ ========================================================================
        test case - rotation in X axis
        ======================================================================== """
        obb3_b.t_ref_obj = torch.tensor([0, 0, 0])
        obb3_b.R_ref_obj = euler_angles_to_matrix(
            [[math.pi / 6, 0, 0]], convention="XYZ"
        )

        self.helper(obb3_a, obb3_b, 1.0, 0.5)

    async def test_distance(self) -> None:
        random_dim, random_T_ref_obj = get_random_dim_and_T(num_sample=20)
        num_a, num_b = 3, 4
        size_a = random_dim[:num_a]
        t_ref_a = random_T_ref_obj[:num_a, :, 3]
        R_ref_a = random_T_ref_obj[:num_a, :, :3]
        instance_id_a = torch.Tensor([1, 2, 3])
        category_a = torch.Tensor([1, 1, 1])

        size_b = random_dim[num_a : num_a + num_b]
        t_ref_b = random_T_ref_obj[num_a : num_a + num_b, :, 3]
        R_ref_b = random_T_ref_obj[num_a : num_a + num_b, :, :3]
        instance_id_b = torch.Tensor([4, 5, 6, 7])
        category_b = torch.Tensor([1, 1, 1, 1])

        obb3_a = Obb3(size_a, t_ref_a, R_ref_a, instance_id_a, category_a)
        obb3_b = Obb3(size_b, t_ref_b, R_ref_b, instance_id_b, category_b)

        pw_dist_chamfer = chamfer_distance(
            obb3_a.bb3_in_ref_frame, obb3_b.bb3_in_ref_frame
        )
        pw_dist_hungarian = hungarian_distance(
            obb3_a.bb3_in_ref_frame, obb3_b.bb3_in_ref_frame
        )

        self.assertEqual(pw_dist_chamfer.shape, (num_a, num_b))
        self.assertEqual(pw_dist_hungarian.shape, (num_a, num_b))

        for i in range(obb3_a.size.shape[0]):
            for j in range(obb3_b.size.shape[0]):
                self.helper(
                    obb3_a, obb3_b, pw_dist_chamfer[i][j], pw_dist_hungarian[i][j], i, j
                )

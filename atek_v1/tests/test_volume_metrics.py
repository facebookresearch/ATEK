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

import unittest

import torch

from atek.evaluation.bbox3d_metrics import iou_giou, iou_giou_single
from atek.tests.test_utils import compute_intersection_volume, get_random_dim_and_T
from atek.utils.obb3 import init_obb3, Obb3


class TestVolumeMetrics(unittest.TestCase):
    def helper(
        self,
        obb3_a: Obb3,
        obb3_b: Obb3,
        gt_intersect_vol: float,
        gt_iou: float,
        gt_giou: float,
        i: int = 0,
        j: int = 0,
    ) -> None:
        vol_in, iou, giou = iou_giou_single(
            obb3_a.size[i], obb3_a.T_ref_obj[i], obb3_b.size[i], obb3_b.T_ref_obj[i]
        )
        vol_in_trimesh = compute_intersection_volume(
            obb3_a.size[j], obb3_a.T_ref_obj[j], obb3_a.size[j], obb3_a.T_ref_obj[j]
        )

        self.assertAlmostEqual(vol_in, vol_in_trimesh, delta=1e-6)
        self.assertAlmostEqual(vol_in, gt_intersect_vol, delta=1e-6)
        self.assertAlmostEqual(iou, gt_iou, delta=1e-6)
        self.assertAlmostEqual(giou, gt_giou, delta=1e-6)

    async def test_iou_giou_single(self) -> None:
        dim_a = [[1, 1, 1]]
        T_a = [[0, 0, 0]]
        angle_a = [[0, 0, 0]]
        instance_id_a = [1]
        category_a = [1]
        obb3_a = init_obb3(
            dim_a, T_a, angle_a, instance_id_a, category_a, convention="XYZ"
        )

        """ ========================================================================
        test case - same box
        ======================================================================== """
        self.helper(obb3_a, obb3_a, 1.0, 1.0, 1.0)

        """ ========================================================================
        test case - scale difference
        ======================================================================== """
        dim_b = [[2, 2, 2]]
        T_b = [[0, 0, 0]]
        angle_b = [[0, 0, 0]]
        instance_id_b = [4]
        category_b = [1]
        obb3_b = init_obb3(
            dim_b, T_b, angle_b, instance_id_b, category_b, convention="XYZ"
        )

        self.helper(obb3_a, obb3_b, 1.0, 0.125, 0.125)

        """ ========================================================================
        test case - scale difference in XY axes
        ======================================================================== """
        obb3_b.size = torch.tensor([[0.5, 2, 1]])

        self.helper(obb3_a, obb3_b, 0.5, 1 / 3, 0.19047619047619035)

        """ ========================================================================
        test case - translation in X axis
        ======================================================================== """
        obb3_b.size = torch.tensor([1, 1, 1])
        obb3_b.t_ref_obj = torch.tensor([0.5, 0, 0])

        self.helper(obb3_a, obb3_b, 0.5, 1 / 3, 1 / 3)

        """ ========================================================================
        test case - translation in X axis (no overlap with one shared face)
        ======================================================================== """
        obb3_b.size = torch.tensor([1, 1, 1])
        obb3_b.t_ref_obj = torch.tensor([1, 0, 0])

        self.helper(obb3_a, obb3_b, 0.0, 0.0, 0.0)

        """ ========================================================================
        test case - translation in XYZ axes (no overlap with one shared vertex)
        ======================================================================== """
        obb3_b.size = torch.tensor([[1, 1, 1]])
        obb3_b.t_ref_obj = torch.tensor([[1, 1, 1]])

        self.helper(obb3_a, obb3_b, 0.0, 0.0, -0.5)

    async def test_iou_giou(self) -> None:
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

        pw_vol_in, pw_iou, pw_giou = iou_giou(
            obb3_a.size, obb3_a.T_ref_obj, obb3_b.size, obb3_b.T_ref_obj
        )

        for i in range(obb3_a.bb3_in_ref_frame.shape[0]):
            for j in range(obb3_b.bb3_in_ref_frame.shape[0]):
                self.helper(
                    obb3_a,
                    obb3_b,
                    pw_vol_in[i][j],
                    pw_iou[i][j],
                    pw_giou[i][j],
                    i,
                    j,
                )

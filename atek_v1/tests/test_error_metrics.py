# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import math
import unittest

from atek.evaluation.bbox3d_metrics import diagonal_error
from atek.evaluation.math_metrics.distance_metrics import euclidean_distance
from atek.evaluation.math_metrics.rotation_metrics import geodesic_angular_error

from pytorch3d.transforms import euler_angles_to_matrix
from torch import tensor

DELTA = 1e-6


class TestErrorMetrics(unittest.TestCase):
    async def test_euclidean_distance(self):
        translation_a = tensor([[0, 0, 0], [0, 1, 0]])
        translation_b = tensor([[1, 0, 0], [1, 1, 1], [1, 2, 3]])
        distance = euclidean_distance(translation_a, translation_b)

        distance_manual = [
            [(x - y).pow(2).sum().sqrt() for y in translation_b] for x in translation_a
        ]
        for i in range(len(translation_a)):
            for j in range(len(translation_b)):
                self.assertAlmostEqual(distance[i][j], distance_manual[i][j], DELTA)

    async def test_geodesic_angular_error(self):
        euler_angle_a = [[0, 0, 0]]
        euler_angle_b = [
            [math.pi / 2, 0, 0],
            [0, math.pi / 6, 0],
            [0, 0, math.pi / 3],
        ]
        rot_mat_a = euler_angles_to_matrix(tensor(euler_angle_a))
        rot_mat_b = euler_angles_to_matrix(tensor(euler_angle_b))
        geodesic_ang_err = geodesic_angular_error(rot_mat_a, rot_mat_b)

        self.assertAlmostEqual(geodesic_ang_err[0, 0], math.pi / 2, DELTA)
        self.assertAlmostEqual(geodesic_ang_err[0, 1], math.pi / 6, DELTA)
        self.assertAlmostEqual(geodesic_ang_err[0, 2], math.pi / 3, DELTA)

    async def test_diagonal_error(self):
        size_a = tensor([[1, 1, 1], [1, 2, 1], [1, 2, 3]])
        size_b = tensor([[1, 0.9, 1], [1, 1.5, 1], [1, 2, 2]])

        diag_error = diagonal_error(size_a, size_b)

        diag_a = [x.pow(2).sum().sqrt() for x in size_a]
        diag_b = [x.pow(2).sum().sqrt() for x in size_b]
        diag_error_manual = [[(x - y).abs() for y in diag_b] for x in diag_a]

        for i in range(len(size_a)):
            for j in range(len(size_b)):
                self.assertAlmostEqual(diag_error[i][j], diag_error_manual[i][j], DELTA)

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

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F


def euclidean_distance(
    pred_translation: torch.Tensor, target_translation: torch.Tensor
) -> torch.Tensor:
    """
    Compute the pairwise Euclidean distance between 2 groups of points in $R^3$.

    Args:
        pred_translation: Tensor of shape (M, 3) representing the predicted translation.
        target_translation: Tensor of shape (N, 3) representing the target translation.

    Returns:
        Tensor of shape (M, N) containing the pairwise Euclidean distance.
    """
    assert pred_translation.ndim == target_translation.ndim == 2
    assert pred_translation.shape[1:] == target_translation.shape[1:]
    M, N = pred_translation.shape[0], target_translation.shape[0]

    # shape: (M, N, 3)
    pred_translation_repeat = pred_translation.unsqueeze(1).repeat((1, N, 1))
    target_translation_repeat = target_translation.unsqueeze(0).repeat((M, 1, 1))

    euclidean_dist = torch.norm(
        pred_translation_repeat - target_translation_repeat, p=2, dim=-1
    )

    return euclidean_dist


def chamfer_distance_single(
    pred: torch.Tensor, target: torch.Tensor, dist_type: F = F.l1_loss
) -> torch.Tensor:
    """
    Compute the chamfer distance between 2 sets of points in $R^d$.

    Args:
        pred: Tensor of shape (..., K, d) representing the predicted values. K and d represent the
            number of points and the dimension of each point.
        target: Tensor of shape (..., K, d) representing the target values. K and d represent the
            number of points and the dimension of each point.
        dist_type: Distance type to use, defaults to F.l1 (L1 distance).

    Returns:
        Tensor of shape (...) containing the chamfer distances.
    """
    assert pred.shape == target.shape
    assert pred.ndim >= 2

    # Calculate the pairwise distances between pred and target
    prefix_ndim = len(pred.shape[:-2])
    K = pred.shape[-2]

    pred_expand_shape = [-1] * prefix_ndim + [-1, K, -1]
    pred_expand = pred.unsqueeze(-2).expand(pred_expand_shape)  # Shape: (..., K, K, d)

    target_expand_shape = [-1] * prefix_ndim + [K, -1, -1]
    target_expand = target.unsqueeze(-3).expand(
        target_expand_shape
    )  # Shape: (..., K, K, d)

    distances = dist_type(pred_expand, target_expand, reduction="none").sum(
        -1
    )  # Shape: (..., K, K)

    # Compute the chamfer distances by taking the minimum distances along each dimension
    pred_min_dist, _ = distances.min(dim=-2)  # Shape: (..., K)
    target_min_dist, _ = distances.min(dim=-1)  # Shape: (..., K)

    # Sum the chamfer distances and return
    chamfer_dist = pred_min_dist.mean(dim=-1) + target_min_dist.mean(
        dim=-1
    )  # Shape: (...)
    return chamfer_dist


def hungarian_distance_single(
    pred: torch.Tensor, target: torch.Tensor, dist_type: F = F.l1_loss
) -> torch.Tensor:
    r"""
    Compute the Hungarian distance: average of the minimum sum of pairwise
    distances between pred and target with 1-1 mapping using Hungarian algorithm.

    Objective: $\min \sum_{i,j} C[i][j] * X[i][j]$, s.t. $\sum_[i] X[i][j] = 1$ and
    $\sum_[j] X[i][j] = 1$, where $C[i][j]$ is the distance between pred[i] and
    target[j], and $X[i][j]$ is the indicator variable for the pair (i, j).

    Args:
        pred: Tensor of shape (K, d) representing the predicted values. K and d represent the
            number of points and the dimension of each point.
        target: Tensor of shape (K, d) representing the target values. K and d represent the
            number of points and the dimension of each point.
        dist_type: Distance type to use, defaults to F.l1 (L1 distance).

    Returns:
        Tensor of shape (1,) containing the Hungarian distances.
    """
    assert pred.ndim == target.ndim == 2
    assert pred.shape == target.shape

    # Calculate the pairwise distances between pred and target
    prefix_ndim = len(pred.shape[:-2])
    K = pred.shape[-2]

    pred_expand_shape = [-1] * prefix_ndim + [-1, K, -1]
    pred_expand = pred.unsqueeze(-2).expand(pred_expand_shape)  # Shape: (K, K, d)

    target_expand_shape = [-1] * prefix_ndim + [K, -1, -1]
    target_expand = target.unsqueeze(-3).expand(target_expand_shape)  # Shape: (K, K, d)

    distances = dist_type(pred_expand, target_expand, reduction="none").sum(
        -1
    )  # Shape: (K, K)

    # use linear assignment to find best 1-1 match
    row_ind, col_ind = linear_sum_assignment(distances)
    dist_hungarian = distances[row_ind, col_ind].mean()

    return dist_hungarian


def chamfer_distance(
    pred: torch.Tensor, target: torch.Tensor, dist_type: F = F.l1_loss
) -> torch.Tensor:
    """
    Compute the pairwise Chamfer distance between 2 sets of points in $R^d$.

    Args:
        pred: Tensor of shape (M, K, d) representing the predicted values. M, K, and d represent
            the number of 3D boxes, the number of points in 3D box, and the dimension of each point.
        target: Tensor of shape (N, K, d) representing the target values. N, K, and d represent the
            number of 3D boxes, the number of points in 3D box, and the dimension of each point.
        dist_type: Distance type to use, defaults to F.l1 (L1 distance).

    Returns:
        Tensor of shape (M, N) containing the pairwise Chamfer distances.
    """
    assert pred.ndim == target.ndim == 3
    assert pred.shape[1:] == target.shape[1:]
    M, N = pred.shape[0], target.shape[0]

    # shape: (M, N, K, d)
    pred_repeat = pred.unsqueeze(1).repeat((1, N, 1, 1))
    target_repeat = target.unsqueeze(0).repeat((M, 1, 1, 1))
    return chamfer_distance_single(pred_repeat, target_repeat)


def hungarian_distance(
    pred: torch.Tensor, target: torch.Tensor, dist_type: F = F.l1_loss, processes=4
) -> torch.Tensor:
    """
    Compute the pairwise Hungarian distance between 2 sets of points in $R^d$.

    Args:
        pred: Tensor of shape (M, K, d) representing the predicted values. M, K, and d represent
            the number of 3D boxes, the number of points in 3D box, and the dimension of each point.
        target: Tensor of shape (N, K, d) representing the target values. N, K, and d represent the
            number of 3D boxes, the number of points in 3D box, and the dimension of each point.
        dist_type: Distance type to use, defaults to F.l1 (L1 distance).
        processes: Number of processes to use for parallelization.

    Returns:
        Tensor of shape (M, N) containing the pairwise Hungarian distances.
    """
    assert pred.ndim == target.ndim == 3
    assert pred.shape[1:] == target.shape[1:]
    M, N = pred.shape[0], target.shape[0]

    # shape: (M, N, K, d)
    pred_repeat = pred.unsqueeze(1).repeat((1, N, 1, 1))
    target_repeat = target.unsqueeze(0).repeat((M, 1, 1, 1))

    # shape: (M * N, K, d)
    pred_repeat = pred_repeat.reshape(-1, pred_repeat.shape[-2], pred_repeat.shape[-1])
    target_repeat = target_repeat.reshape(
        -1, target_repeat.shape[-2], target_repeat.shape[-1]
    )

    # TODO: add batched implementation of this function to accelerate
    pairwise_dist = []
    for prd, tgt in zip(pred_repeat, target_repeat):
        dist_hung = hungarian_distance_single(prd, tgt, dist_type)
        pairwise_dist.append(dist_hung)

    # shape: (M, N)
    pairwise_dist_hungarian = torch.tensor(pairwise_dist).reshape(M, N)

    return pairwise_dist_hungarian

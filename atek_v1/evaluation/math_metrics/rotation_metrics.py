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


def batch_geodesic_loss(
    m1: torch.Tensor, m2: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    m1, m2: B x 3 x 3 rotation matrix
    """
    assert m1.ndim == 3
    assert m2.ndim == 3
    assert m1.shape[0] == m2.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2

    return torch.acos(torch.clamp(cos, -1 + eps, 1 - eps))


def geodesic_angular_error(
    pred_rotation: torch.Tensor, target_rotation: torch.Tensor
) -> torch.Tensor:
    """
    Compute the pairwise Geodesic angular error between pred_rotation and target_rotation.

    Args:
        pred_rotation: Tensor of shape (M, 3, 3) representing the predicted rotation.
        target_rotation: Tensor of shape (N, 3, 3) representing the target rotation.

    Returns:
        Tensor of shape (M, N) containing the pairwise Geodesic angular error.
    """
    assert pred_rotation.ndim == target_rotation.ndim == 3
    assert pred_rotation.shape[1:] == target_rotation.shape[1:]
    assert pred_rotation.shape[1:] == (3, 3)

    M, N = pred_rotation.shape[0], target_rotation.shape[0]

    # shape: (M, N, 3, 3)
    pred_rotation_repeat = pred_rotation.unsqueeze(1).repeat((1, N, 1, 1))
    target_rotation_repeat = target_rotation.unsqueeze(0).repeat((M, 1, 1, 1))

    # shape: (M * N, 3, 3)
    pred_rotation_repeat = pred_rotation_repeat.reshape(
        -1, pred_rotation_repeat.shape[-2], pred_rotation_repeat.shape[-1]
    )
    target_rotation_repeat = target_rotation_repeat.reshape(
        -1, target_rotation_repeat.shape[-2], target_rotation_repeat.shape[-1]
    )

    # shape: (M, N)
    geodesic_angular_error = batch_geodesic_loss(
        pred_rotation_repeat, target_rotation_repeat
    ).reshape(M, N)

    return geodesic_angular_error

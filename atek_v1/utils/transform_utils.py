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


def get_cuboid_corners(half_extents: torch.Tensor) -> torch.Tensor:
    """
    https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/iou_box3d.py#L111)

    (4) +---------+. (5)
        | ` .     |  ` .
        | (0) +---+-----+ (1)
        |     |   |     |
    (7) +-----+---+. (6)|
        ` .   |     ` . |
        (3) ` +---------+ (2)
    NOTE: Throughout this implementation, we assume that boxes
    are defined by their 8 corners exactly in the order specified in the
    diagram above for the function to give correct results. In addition
    the vertices on each plane must be coplanar.
    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

    box_corner_vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]

    Args:
        half_extents: Half extents of the cuboid in the format of (N, 3)

    Returns:
        corners: Corners of the cuboid in the format of (N, 8, 3)
    """
    corners = torch.tensor(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=torch.float32,
        device=half_extents.device,
    )

    # Scale the corners of the unit box
    corners = corners * half_extents.unsqueeze(1)

    return corners


def batch_transform_points(
    points_in_B: torch.tensor, T_A_B: torch.Tensor
) -> torch.Tensor:
    """
    Return point_in_A = R_A_B @ points_in_B + t_A_B in shape [N x M x 3]
    Args:
        points_in_B (torch.Tensor): NxMx3 tensor for a batch N of M 3D points in frame B,
          corresponding to each transformation matrix
        T_A_B (torch.Tensor): Nx3x4 transformation matrices from B to A

    Returns:
        points_in_A (torch.Tensor): NxMx3 tensor of a batch N of M 3D points in frame A, after
            transformation
    """

    # Reshape points to (N, 3, M) for batch matrix multiplication
    points_in_B_reshaped = points_in_B.permute(0, 2, 1)
    M = points_in_B_reshaped.shape[-1]

    R_A_B = T_A_B[:, :, :3]
    t_A_B = T_A_B[:, :, 3]
    points_in_A = (
        torch.bmm(R_A_B, points_in_B_reshaped) + t_A_B.unsqueeze(-1).repeat(1, 1, M)
    ).permute(0, 2, 1)

    return points_in_A


def batch_inverse(T: torch.Tensor) -> torch.Tensor:
    """
    Inverse SE3 transformation matrix T in Tensor of shape (..., 3, 4).

    Args:
        T: (..., 3, 4) tensor, transformation matrix

    Returns:
        inversed_T: (..., 3, 4) tensor, inverse transformation matrix
    """
    inversed_T = T.clone()
    inversed_T[..., :, :3] = T[..., :, :3].transpose(1, 2)
    inversed_T[..., :, 3] = (
        -inversed_T[..., :, :3] @ (T[..., :, 3].unsqueeze(2))
    ).squeeze()
    return inversed_T

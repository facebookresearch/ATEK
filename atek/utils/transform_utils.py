# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

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
    points_in_B: torch.tensor,
    R_A_B: torch.tensor,
    t_A_B: torch.tensor = None,
):
    """
    Return point_in_A = R_A_B @ points_in_B + t_A_B in shape [N x M x 3]
    Args:
        points_in_B : NxMx3  M points coorsponds to each rotation matrix
        R_A_B: Nx3x3  N rotations matrices from B to A
        t_A_B: Nx3  N translation from B to A
    """

    # Reshape points to (N, 3, M) for batch matrix multiplication
    points_in_B_reshaped = points_in_B.permute(0, 2, 1)
    M = points_in_B_reshaped.shape[-1]

    points_in_A = torch.bmm(R_A_B, points_in_B_reshaped)

    if t_A_B is not None:
        points_in_A += t_A_B.unsqueeze(-1).repeat(1, 1, M)

    return points_in_A.permute(0, 2, 1)

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

from typing import Tuple, Union

import numpy as np
import torch

from atek_v1.utils.transform_utils import (
    batch_inverse,
    batch_transform_points,
    get_cuboid_corners,
)
from scipy.spatial import ConvexHull


BBOX_LINES_3D = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]

BBOX_FACES_3D = [
    [4, 5, 6, 7],  # top face zmax
    [0, 1, 2, 3],  # bottom face zmin
    [2, 3, 7, 6],  # left face ymax
    [0, 1, 5, 4],  # right face ymin
    [1, 2, 6, 5],  # front face xmax
    [3, 0, 4, 7],  # back face xmin
]


def is_point_on_line_segment(point, start_point, end_point):
    """
    Check if a 3D point lies on a line segment defined by two other 3D points.
    """
    # Calculate the vectors from the start point to the end point and from the start point to the test point
    vector1 = end_point - start_point
    vector2 = point - start_point

    # Check if the cross product of the two vectors is approximately zero
    cross_product = torch.cross(vector1, vector2)
    if torch.allclose(cross_product, torch.zeros(3, dtype=point.dtype)):
        # Check if the dot product of the two vectors is positive and less than the squared length of vector1
        dot_product = torch.dot(vector1, vector2)
        if 0 <= dot_product <= torch.dot(vector1, vector1):
            return True

    return False


def is_point_inside_triangle(p, a, b, c):
    """
    Check if a 3D point lies inside a triangle defined by three other 3D points.
    """
    # Compute vectors
    v0 = c - a
    v1 = b - a
    v2 = p - a

    # Compute dot products
    dot00 = torch.dot(v0, v0)
    dot01 = torch.dot(v0, v1)
    dot02 = torch.dot(v0, v2)
    dot11 = torch.dot(v1, v1)
    dot12 = torch.dot(v1, v2)

    # Compute barycentric coordinates
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    # Check if the point is in the triangle
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def is_point_inside_plane(intersection_point, plane_points):
    """
    Check if a point lies inside a plane defined by four points.
    """
    if (
        is_point_inside_triangle(
            intersection_point, plane_points[0], plane_points[1], plane_points[2]
        )
        or is_point_inside_triangle(
            intersection_point, plane_points[0], plane_points[1], plane_points[3]
        )
        or is_point_inside_triangle(
            intersection_point, plane_points[0], plane_points[2], plane_points[3]
        )
        or is_point_inside_triangle(
            intersection_point, plane_points[3], plane_points[1], plane_points[2]
        )
        or is_point_on_line_segment(
            intersection_point, plane_points[0], plane_points[1]
        )
        or is_point_on_line_segment(
            intersection_point, plane_points[1], plane_points[2]
        )
        or is_point_on_line_segment(
            intersection_point, plane_points[2], plane_points[3]
        )
        or is_point_on_line_segment(
            intersection_point, plane_points[3], plane_points[0]
        )
    ):
        return True
    return False


def line_plane_intersection(
    plane_points: torch.Tensor, line_points: torch.Tensor
) -> Union[None, torch.Tensor]:
    """
    Compute the intersection point of a line and a plane.

    Args:
        plane_points: (4, 3) tensor, four points on the plane
        line_points: (2, 3) tensor, two points on the line

    Returns:
        intersection_point: (3,) tensor, intersection point
    """
    assert plane_points.size() == (4, 3)
    assert line_points.size() == (2, 3)

    # Compute the normal of the plane
    normal = torch.cross(
        plane_points[1] - plane_points[0], plane_points[2] - plane_points[0]
    )
    D = -torch.dot(normal, plane_points[0])

    # Compute the intersection point of line and plane
    P1, P2 = line_points[0], line_points[1]
    t = -(torch.dot(normal, P1) + D) / (torch.dot(normal, P2 - P1) + 1e-9)
    if 0 <= t.item() <= 1:
        intersection_point = P1 + t * (P2 - P1)

        # Check if point is inside plane
        if is_point_inside_plane(intersection_point, plane_points):
            return intersection_point

    return None


def check_coplanar(points: np.ndarray) -> bool:
    # check if the points are coplanar
    rank = np.linalg.matrix_rank(points - np.expand_dims(points[0], 0))
    return rank < 3


def iou_giou_single(
    cuboid_A_dimensions: torch.Tensor,
    T_world_A: torch.Tensor,
    cuboid_B_dimensions: torch.Tensor,
    T_world_B: torch.Tensor,
) -> Tuple[float, float, float]:
    """
    Computes IOU and GIOU between two 3D bounding boxes.

    Args:
        cuboid_A_dimensions: (3,) tensor, dimension of cuboid A
        T_world_A: (3, 4) tensor, transformation matrix from object to world of cuboid A
        cuboid_B_dimensions: (3,) tensor, dimension of cuboid B
        T_world_B: (3, 4) tensor, transformation matrix from object to world of cuboid B

    Returns:
        vol_in: float, volume of intersection
        iou: float, intersection over union
        giou: float, generalized intersection over union
    """
    assert cuboid_A_dimensions.shape == (3,)
    assert T_world_A.shape == (3, 4)
    assert cuboid_B_dimensions.shape == (3,)
    assert T_world_B.shape == (3, 4)

    half_extents_A = cuboid_A_dimensions.unsqueeze(0) / 2
    half_extents_B = cuboid_B_dimensions.unsqueeze(0) / 2

    cuboid_A_corners_in_world = batch_transform_points(
        get_cuboid_corners(half_extents_A),
        T_world_A.unsqueeze(0),
    )
    cuboid_B_corners_in_world = batch_transform_points(
        get_cuboid_corners(half_extents_B),
        T_world_B.unsqueeze(0),
    )

    # compute convex hull for A and B
    convex_hull_A = ConvexHull(cuboid_A_corners_in_world.squeeze(0).numpy())
    convex_hull_B = ConvexHull(cuboid_B_corners_in_world.squeeze(0).numpy())

    # compute convex hull for all points from A and B
    all_points = np.row_stack(
        [
            cuboid_A_corners_in_world.squeeze(0).numpy(),
            cuboid_B_corners_in_world.squeeze(0).numpy(),
        ]
    )
    convex_hull_all = ConvexHull(all_points)

    # find all the intersection points of A and B
    points_for_intersection = []

    cuboid_A_corners_in_B = batch_transform_points(
        cuboid_A_corners_in_world,
        batch_inverse(T_world_B.unsqueeze(0)),
    )
    cuboid_B_corners_in_A = batch_transform_points(
        cuboid_B_corners_in_world,
        batch_inverse(T_world_A.unsqueeze(0)),
    )

    A_inside_B_idx = (
        (-half_extents_B < cuboid_A_corners_in_B)
        & (cuboid_A_corners_in_B < half_extents_B)
    ).all(dim=-1)
    cuboid_A_corners_inside_B_in_world = cuboid_A_corners_in_world[A_inside_B_idx]
    if len(cuboid_A_corners_inside_B_in_world) > 0:
        points_for_intersection.append(cuboid_A_corners_inside_B_in_world.numpy())

    B_inside_A_idx = (
        (-half_extents_A < cuboid_B_corners_in_A)
        & (cuboid_B_corners_in_A < half_extents_A)
    ).all(dim=-1)
    cuboid_B_corners_inside_A_in_world = cuboid_B_corners_in_world[B_inside_A_idx]
    if len(cuboid_B_corners_inside_A_in_world) > 0:
        points_for_intersection.append(cuboid_B_corners_inside_A_in_world.numpy())

    edge_intersection_A_in_world = []
    edge_intersection_B_in_world = []
    for LINE in BBOX_LINES_3D:
        for FACE in BBOX_FACES_3D:
            plane_points_A_in_world = cuboid_A_corners_in_world.squeeze()[FACE]
            plane_points_B_in_world = cuboid_B_corners_in_world.squeeze()[FACE]

            line_points_A_in_world = cuboid_A_corners_in_world.squeeze()[LINE]
            line_points_B_in_world = cuboid_B_corners_in_world.squeeze()[LINE]

            intersect_points_in_world_A = line_plane_intersection(
                plane_points_B_in_world, line_points_A_in_world
            )
            if intersect_points_in_world_A is not None:
                edge_intersection_A_in_world.append(intersect_points_in_world_A)

            intersect_points_in_world_B = line_plane_intersection(
                plane_points_A_in_world, line_points_B_in_world
            )
            if intersect_points_in_world_B is not None:
                edge_intersection_B_in_world.append(intersect_points_in_world_B)

    if len(edge_intersection_A_in_world) > 0:
        points_for_intersection.append(
            torch.stack(edge_intersection_A_in_world).numpy()
        )
    if len(edge_intersection_B_in_world) > 0:
        points_for_intersection.append(
            torch.stack(edge_intersection_B_in_world).numpy()
        )

    # compute convex hull for intersection of A and B
    # if there are no intersection points, then the volume is zero
    if len(points_for_intersection) > 0:
        all_intersection_points_in_world = np.row_stack(points_for_intersection)

        # if all points are coplanar, then the volume is zero
        if check_coplanar(np.unique(all_intersection_points_in_world, axis=0)):
            vol_in = 0
        else:
            ch_intersect = ConvexHull(all_intersection_points_in_world)
            vol_in = ch_intersect.volume
    else:
        vol_in = 0

    # compute volume of union of A and B, IoU, GIoU
    AUB = convex_hull_A.volume + convex_hull_B.volume - vol_in
    iou = vol_in / AUB
    giou = iou - (convex_hull_all.volume - AUB) / convex_hull_all.volume

    return vol_in, iou, giou


def iou_giou(
    cuboid_A_dimensions: torch.Tensor,
    T_world_A: torch.Tensor,
    cuboid_B_dimensions: torch.Tensor,
    T_world_B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes pairwise IOU and GIOU between M and N 3D bounding boxes.

    Args:
        cuboid_A_dimensions: (M, 3) tensor, dimension of cuboids A
        T_world_A: (M, 3, 4) tensor, transformation matrices from object to world of cuboids A
        cuboid_B_dimensions: (N, 3) tensor, dimension of cuboids B
        T_world_B: (N, 3, 4) tensor, transformation matrices from object to world of cuboids B

    Returns:
        pw_vol_in: (M, N) tensor, pairwise volumes of intersections
        pw_iou: (M, N) tensor, pairwise IOUs
        pw_giou: (M, N) tensor, pairwise GIOUs
    """

    assert cuboid_A_dimensions.ndim == cuboid_B_dimensions.ndim == 2
    assert cuboid_A_dimensions.shape[1] == cuboid_B_dimensions.shape[1] == 3
    assert T_world_A.ndim == T_world_B.ndim == 3
    assert T_world_A.shape[1:] == T_world_B.shape[1:] == (3, 4)

    M, N = len(cuboid_A_dimensions), len(cuboid_B_dimensions)
    dim_A_repeat = cuboid_A_dimensions.unsqueeze(1).repeat(1, N, 1)
    dim_B_repeat = cuboid_B_dimensions.unsqueeze(0).repeat(M, 1, 1)
    T_world_A_repeat = T_world_A.unsqueeze(1).repeat(1, N, 1, 1)
    T_world_B_repeat = T_world_B.unsqueeze(0).repeat(M, 1, 1, 1)

    dim_A_repeat = dim_A_repeat.reshape((M * N, 3))
    dim_B_repeat = dim_B_repeat.reshape((M * N, 3))
    T_world_A_repeat = T_world_A_repeat.reshape((M * N, 3, 4))
    T_world_B_repeat = T_world_B_repeat.reshape((M * N, 3, 4))

    # TODO: add batched implementation of this function to accelerate
    pw_vol_in, pw_iou, pw_giou = [], [], []
    for dim_A, T_A, dim_B, T_B in zip(
        dim_A_repeat, T_world_A_repeat, dim_B_repeat, T_world_B_repeat
    ):
        vol_in, iou, giou = iou_giou_single(dim_A, T_A, dim_B, T_B)
        pw_vol_in.append(vol_in)
        pw_iou.append(iou)
        pw_giou.append(giou)
    pw_vol_in = torch.Tensor(pw_vol_in).reshape((M, N))
    pw_iou = torch.Tensor(pw_iou).reshape((M, N))
    pw_giou = torch.Tensor(pw_giou).reshape((M, N))

    return pw_vol_in, pw_iou, pw_giou


def diagonal_error(
    pred_scale: torch.Tensor, target_scale: torch.Tensor
) -> torch.Tensor:
    """
    Compute the diagonal error between pred_scale and target_scale.

    Args:
        pred_scale: Tensor of shape (M, 3) representing the predicted scale.
        target_scale: Tensor of shape (N, 3) representing the target scale.

    Returns:
        Tensor of shape (M, N) containing the pairwise diagonal error.
    """
    assert pred_scale.ndim == target_scale.ndim == 2
    assert pred_scale.shape[1:] == target_scale.shape[1:]

    M, N = pred_scale.shape[0], target_scale.shape[0]

    # shape: (M, N, 3)
    pred_scale_repeat = pred_scale.unsqueeze(1).repeat((1, N, 1))
    target_scale_repeat = target_scale.unsqueeze(0).repeat((M, 1, 1))

    # shape: (M, N)
    pred_diag = torch.norm(pred_scale_repeat, p=2, dim=-1)
    target_diag = torch.norm(target_scale_repeat, p=2, dim=-1)

    diag_error = torch.abs(pred_diag - target_diag)

    return diag_error

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


# pyre-strict

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from atek.util.tensor_utils import compute_bbox_corners_in_world
from projectaria_tools.core.calibration import CameraProjection
from projectaria_tools.core.sophus import SE3

COLOR_GREEN = [30, 255, 30]
COLOR_RED = [255, 30, 30]
COLOR_BLUE = [30, 30, 255]
COLOR_GRAY = [200, 200, 200, 100]


def box_points_to_lines(projected_box) -> list[list[float]]:
    """
    Convert a list of 8 points to a list of lines.
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = (
        projected_box[0],
        projected_box[1],
        projected_box[2],
        projected_box[3],
        projected_box[4],
        projected_box[5],
        projected_box[6],
        projected_box[7],
    )
    return [
        [p1, p2],
        [p2, p3],
        [p3, p4],
        [p4, p1],
        [p5, p6],
        [p6, p7],
        [p7, p8],
        [p8, p5],
        [p1, p5],
        [p2, p6],
        [p3, p7],
        [p4, p8],
    ]


def _sample_points_on_3D_line(start, end, num_samples=10) -> List:
    points = [
        start + t * (end - start) for t in np.linspace(0, 1, num_samples)
    ]  # List [tensor(3)]
    point_pairs = [
        (points[i], points[i + 1]) for i in range(len(points) - 1)
    ]  # List[(tensor(3), tensor(3))]
    return point_pairs


def box_points_to_segmented_edges(box_corners: List, num_segments: int = 10) -> List:
    """
    Convert the 8 corners a 3D bounding box, to a list of sampled consecutive points on the 12 edges
    Return 12 lists of point pairs, each representing an edge
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = (
        box_corners[0],
        box_corners[1],
        box_corners[2],
        box_corners[3],
        box_corners[4],
        box_corners[5],
        box_corners[6],
        box_corners[7],
    )
    return [
        _sample_points_on_3D_line(p1, p2, num_segments),
        _sample_points_on_3D_line(p2, p3, num_segments),
        _sample_points_on_3D_line(p3, p4, num_segments),
        _sample_points_on_3D_line(p4, p1, num_segments),
        _sample_points_on_3D_line(p5, p6, num_segments),
        _sample_points_on_3D_line(p6, p7, num_segments),
        _sample_points_on_3D_line(p7, p8, num_segments),
        _sample_points_on_3D_line(p8, p5, num_segments),
        _sample_points_on_3D_line(p1, p5, num_segments),
        _sample_points_on_3D_line(p2, p6, num_segments),
        _sample_points_on_3D_line(p3, p7, num_segments),
        _sample_points_on_3D_line(p4, p8, num_segments),
    ]


def check_projected_points_within_image(
    projected_points: List[np.ndarray], image_width: int, image_height: int
) -> bool:
    """
    Check if the projected points are completely within the image.
    """
    complete_in_of_image = True
    for point in projected_points:
        # Found a point inside the image, return False
        if (
            point[0] < -0.5
            or point[0] > image_width - 0.5
            or point[1] < -0.5
            or point[1] > image_height
        ):
            return False
    return True


def filter_line_segs_out_of_camera_view(
    line_segs: List[List[Tuple[np.ndarray, np.ndarray]]],
    camera_projection: CameraProjection,
    T_World_Camera: SE3,
    image_width: int,
    image_height: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    filtered_line_segs = []
    for edge in line_segs:
        filtered_edge = []
        for seg in edge:
            start_point_in_cam = T_World_Camera.inverse() @ seg[0]
            end_point_in_cam = T_World_Camera.inverse() @ seg[1]

            # Remove if any point is behind the camera
            if start_point_in_cam[2] < 0 or end_point_in_cam[2] < 0:
                continue

            # Project the start and end points
            projected_start = camera_projection.project(start_point_in_cam)
            projected_end = camera_projection.project(end_point_in_cam)

            # Check if either points are out of the image
            if not check_projected_points_within_image(
                [projected_start, projected_end], image_width, image_height
            ):
                continue

            filtered_edge.append([projected_start, projected_end])
        filtered_line_segs.append(filtered_edge)
    return filtered_line_segs


def obtain_visible_line_segs_of_obb3(
    obb3_corners_in_world,
    camera_projection: CameraProjection,
    T_World_Camera: SE3,
    image_width: int,
    image_height: int,
) -> Tuple[List, List]:
    # Sample the points on the 12 edges, returns a list of 12
    line_segs_on_edges = box_points_to_segmented_edges(obb3_corners_in_world)

    # Filter out the line segments that are not visible in the camera view
    visible_line_segs = filter_line_segs_out_of_camera_view(
        line_segs=line_segs_on_edges,
        camera_projection=camera_projection,
        T_World_Camera=T_World_Camera,
        image_width=image_width,
        image_height=image_height,
    )

    # Aggreagate visible line segments along with colors
    all_visible_line_segs = []
    all_seg_colors = []
    plotting_colors = [
        COLOR_RED,
        COLOR_GRAY,
        COLOR_GRAY,
        COLOR_GREEN,
        COLOR_GRAY,
        COLOR_GRAY,
        COLOR_GRAY,
        COLOR_GRAY,
        COLOR_BLUE,
        COLOR_GRAY,
        COLOR_GRAY,
        COLOR_GRAY,
    ]
    for i_edge in range(12):
        for seg in visible_line_segs[i_edge]:
            all_visible_line_segs.append(seg)
            all_seg_colors.append(plotting_colors[i_edge])

    return all_visible_line_segs, all_seg_colors

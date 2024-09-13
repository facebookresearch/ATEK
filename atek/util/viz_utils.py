# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import Dict, List, Optional

import numpy as np
import torch


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


def check_projected_points_out_of_image(
    projected_points: List[np.ndarray], image_width: int, image_height: int
) -> bool:
    """
    Check if the projected points are completely out of the image.
    Return True if all points are out of the image.
    """
    complete_out_of_image = True
    for point in projected_points:
        # Found a point inside the image, return False
        if (
            point[0] > -0.5
            and point[0] <= image_width - 0.5
            and point[1] > -0.5
            and point[1] <= image_height
        ):
            return False
    return True


def filter_obbs_by_confidence(
    obb_dict: Dict, confidence_score: torch.Tensor, confidence_lower_threshold: float
) -> Dict:
    """
    A simple filter function to filter the predictions by confidence score (lower_threshold).
    The obb_dict follows the format of ATEK obb3_gt_processor convention
    """
    filtered_indices = confidence_score > confidence_lower_threshold
    result_dict = obb_dict.copy()  # shallow copy, should be okay

    for camera_label, single_cam_dict in obb_dict.items():
        for key, val in single_cam_dict.items():
            if isinstance(val, torch.Tensor):
                result_dict[camera_label][key] = val[filtered_indices]
            if isinstance(val, List):
                result_dict[camera_label][key] = [
                    s
                    for s, selected_flag in zip(val, filtered_indices)
                    if selected_flag
                ]

    return result_dict

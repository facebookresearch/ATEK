# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List

import torch
from projectaria_tools.core.calibration import CameraCalibration


def undistort_pixel_coords(
    pixels: torch.Tensor,
    src_calib: CameraCalibration,
    dst_calib: CameraCalibration,
) -> torch.Tensor:
    """
    A function to batch undistort pixel coords (tensor [N, 2]) from src_calib to dst_calib.
    """
    new_pixel_list = []
    for i in range(pixels.shape[1]):
        unprojected_ray = src_calib.unproject_no_checks(pixels[:, i].numpy())
        new_pixel_list.append(
            torch.tensor(dst_calib.project_no_checks(unprojected_ray))
        )
    return torch.stack(new_pixel_list)


def rescale_pixel_coords(pixels: torch.Tensor, scale: float) -> torch.Tensor:
    """
    A function to batch rescale pixel coords (tensor [N, 2]) by a scale factor.
    """
    return (pixels - 0.5) * scale + 0.5


def rotate_pixel_coords_cw90(
    pixels: torch.Tensor, image_dim_after_rot: List
) -> torch.Tensor:
    """
    batch rotate pixel coords by 90deg clockwise, where pixels is a tensor of [N, 2]
    """
    # this looks like swapped because it is easier to pass in dim_after_rotation instead of dim_before_rotation
    center_y = (image_dim_after_rot[0] - 1.0) / 2.0
    center_x = (image_dim_after_rot[1] - 1.0) / 2.0

    translated_pixels = pixels - torch.tensor([[center_x, center_y]])
    rotated_pixels = torch.stack(
        [translated_pixels[:, 1], -translated_pixels[:, 0]], dim=1
    )
    rotated_pixels += torch.tensor([[center_x, center_y]])

    return rotated_pixels

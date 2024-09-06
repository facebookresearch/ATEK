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

import logging
from typing import Union

import numpy as np
import torch
import trimesh
from projectaria_tools.core import calibration

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def linear_cam_matrix_from_intrinsics(
    camera_params_fufvu0v0: Union[torch.Tensor, np.ndarray]
) -> Union[torch.Tensor, np.ndarray]:
    """
    Generates the camera matrix (nx3x3) for n pinhole camera parameters n x (fx, fy, cx, cy)
    """
    assert camera_params_fufvu0v0.ndim == 2
    assert camera_params_fufvu0v0.shape[-1] == 4
    num_cameras = camera_params_fufvu0v0.shape[0]
    # Initialize an array of zeros
    if isinstance(camera_params_fufvu0v0, torch.Tensor):
        camera_matrices = torch.zeros((num_cameras, 3, 3))
    else:
        camera_matrices = np.zeros((num_cameras, 3, 3))

    # Assign fx, fy, cx, cy
    camera_matrices[:, 0, 0] = camera_params_fufvu0v0[:, 0]  # fx
    camera_matrices[:, 1, 1] = camera_params_fufvu0v0[:, 1]  # fy
    camera_matrices[:, 0, 2] = camera_params_fufvu0v0[:, 2]  # cx
    camera_matrices[:, 1, 2] = camera_params_fufvu0v0[:, 3]  # cy
    camera_matrices[:, 2, 2] = 1.0

    return camera_matrices


def get_camera_fov_cone_angle(
    camera_model: calibration.CameraCalibration,
):
    """Compute the angle of the camera FOV cone in rad."""
    W, H = camera_model.get_image_size()
    principal_point = camera_model.get_principal_point()
    center_ray = np.array([0, 0, 1])

    # Define 4 extremes points to compute the max angle of the camera FOV cone.
    extreme_points = np.array(
        [
            [principal_point[0], 0],
            [principal_point[0], W - 1],
            [0, principal_point[1]],
            [H - 1, principal_point[1]],
        ]
    )
    extreme_points_ray = np.array(
        [
            camera_model.unproject_no_checks(extreme_point)
            for extreme_point in extreme_points
        ]
    )

    # Normalize the extreme points rays
    normalized_rays = extreme_points_ray / np.linalg.norm(
        extreme_points_ray, axis=1, keepdims=True
    )

    # Compute the angles using the dot product and arccos, then convert from radians to degrees
    half_cone_angles = np.arccos(np.dot(normalized_rays, center_ray))

    return half_cone_angles.max()


def create_spherical_cone(
    radius: float, angle: float, circle_segments: int = 32, cap_segments: int = 3
):
    """
    Helper function to create a spherical cone with the given radius and angle.
    Note the cone point is always at the 0, 0, 0, and the cone is aligned with
    the +z axis.
    """
    theta = np.linspace(0, 2 * np.pi, circle_segments)
    phi = np.linspace(0, angle, cap_segments)

    theta, phi = np.meshgrid(theta, phi)

    # Adjust the coordinates to match the desired cone point and angle
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Create vertices from the coordinate arrays
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = np.row_stack([np.array([0, 0, 0]), vertices])

    mesh = trimesh.convex.convex_hull(vertices)

    return mesh


def get_camera_fov_spherical_cone(
    camera_model: calibration.CameraCalibration,
    far_clipping_distance: float = 4.0,
    circle_segments: int = 32,
    cap_segments: int = 3,
):
    """
    Helper function to create a spherical cone that represents
    the camera FOV cone. The cone is centered on the camera optical axis.
    """
    half_cone_angle = get_camera_fov_cone_angle(camera_model)
    return create_spherical_cone(
        far_clipping_distance, half_cone_angle, circle_segments, cap_segments
    )

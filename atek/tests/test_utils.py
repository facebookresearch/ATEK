import math
from typing import Callable, Tuple

import numpy as np
import torch
import trimesh

from atek.utils.mesh_boolean_utils import intersect_meshes
from atek.utils.transform_utils import batch_transform_points, get_cuboid_corners

from pytorch3d.transforms import euler_angles_to_matrix
from torch import tensor


def get_random_dim_and_T(
    num_sample: int = 10,
    max_scale_factor: float = 1.1,
    max_translation_factor: float = 0.1,
    max_rotation_radian: float = math.pi / 6,
    rand_fn: Callable = torch.rand,
    random_seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get some random dimensions and transformation matrices

    Args:
        num_sample: number of samples to generate.
        max_scale_factor: maximum scale factor to perturb object box dimensions.
        max_translation_factor: maximum translation factor to perturb object box center.
        max_rotation_radian: maximum rotation radian to perturb object box rotation.
        rand_fn: random function to use.
        random_seed: random seed to use.

    Returns:
        perturbed_dimensions: (num_sample, 3) tensor of perturbed object dimensions
        perturbed_T_ref_obj: (num_sample, 3, 4) tensor of perturbed transformation matrices from
            reference to object coordinate
    """
    dimension = tensor([1, 1, 1])
    translation = tensor([0, 0, 0])
    rotation_angle = tensor([0, 0, 0])
    R_ref_obj = euler_angles_to_matrix(rotation_angle, convention="XYZ")
    T_ref_obj = torch.cat((R_ref_obj, translation.unsqueeze(1)), dim=1)

    # Set the random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    min_scale_factor = 1 / max_scale_factor
    assert max_scale_factor >= min_scale_factor

    # perturb dimensions
    scale_factors = []
    for _ in [0, 1, 2]:
        scale_factors.append(
            rand_fn(num_sample, 1) * (max_scale_factor - min_scale_factor)
            + min_scale_factor
        )
    scale_factors = torch.cat(scale_factors, dim=1)
    perturbed_dimensions = dimension.repeat(num_sample, 1) * scale_factors

    # perturb translation
    translation_offsets = []
    for _ in [0, 1, 2]:
        translation_offsets.append(
            (rand_fn(num_sample, 1) * 2 - 1) * max_translation_factor
        )
    translation_offsets = torch.cat(translation_offsets, dim=1)
    pertubed_translations = T_ref_obj[:, 3].repeat(
        num_sample, 1
    ) + translation_offsets * dimension.repeat(num_sample, 1)

    # perturb rotation
    euler_angles_offsets = []
    for _ in [0, 1, 2]:
        euler_angles_offsets.append(
            (rand_fn(num_sample, 1) * 2 - 1) * max_rotation_radian
        )
    euler_angles_offsets = torch.cat(euler_angles_offsets, dim=1)
    rotation_offsets = euler_angles_to_matrix(euler_angles_offsets, convention="XYZ")
    pertubed_rotations = T_ref_obj[:, :3].repeat(num_sample, 1, 1) @ rotation_offsets

    perturbed_T_ref_obj = torch.cat(
        [pertubed_rotations, pertubed_translations.unsqueeze(-1)], dim=-1
    )

    return perturbed_dimensions, perturbed_T_ref_obj


def compute_intersection_volume(
    cuboid_A_dimensions: torch.Tensor,
    T_world_A: torch.Tensor,
    cuboid_B_dimensions: torch.Tensor,
    T_world_B: torch.Tensor,
) -> float:
    """
    Computes volume of intersection between two cuboids, using trimesh for unit tests.

    Args:
        cuboid_A_dimensions: (3,) tensor, dimension of cuboid A
        T_world_A: (3, 4) tensor, transformation matrix from object to world of cuboid A
        cuboid_B_dimensions: (3,) tensor, dimension of cuboid B
        T_world_B: (3, 4) tensor, transformation matrix from object to world of cuboid B

    Returns:
        vol_in: float, volume of intersection between cuboid A and B
    """
    assert cuboid_A_dimensions.shape == (3,)
    assert T_world_A.shape == (3, 4)
    assert cuboid_B_dimensions.shape == (3,)
    assert T_world_B.shape == (3, 4)

    half_extents_A = cuboid_A_dimensions.unsqueeze(0) / 2
    half_extents_B = cuboid_B_dimensions.unsqueeze(0) / 2

    cuboid_A_corners_in_obj = get_cuboid_corners(half_extents_A)
    cuboid_A_corners_in_world = batch_transform_points(
        cuboid_A_corners_in_obj,
        T_world_A[:, :3].unsqueeze(0),
        T_world_A[:, 3].unsqueeze(0),
    )

    cuboid_B_corners_in_obj = get_cuboid_corners(half_extents_B)
    cuboid_B_corners_in_world = batch_transform_points(
        cuboid_B_corners_in_obj,
        T_world_B[:, :3].unsqueeze(0),
        T_world_B[:, 3].unsqueeze(0),
    )

    # compute intersection
    trimesh_a = trimesh.convex.convex_hull(cuboid_A_corners_in_world.squeeze(0).numpy())
    trimesh_b = trimesh.convex.convex_hull(cuboid_B_corners_in_world.squeeze(0).numpy())
    inter = intersect_meshes(trimesh_a, trimesh_b)
    vol_in = float(inter.volume)

    return vol_in

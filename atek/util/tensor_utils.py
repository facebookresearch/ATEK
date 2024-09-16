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

from typing import Dict, List, Optional, Tuple

import numpy as np

import torch
from projectaria_tools.core.sophus import SE3


def fill_or_trim_tensor(tensor: torch.Tensor, dim_size: int, dim: int, fill_value=None):
    """Fill or trim a torch tensor to the given `dim_size`, along the given dim

        Inputs:
            tensor (torch tensor): input tensor
            dim_size (int): the size to fill or trim to (e.g. predefined batch size)
            dim (int): the dimension to fill or trim
            fill_value: if set, fill the tensor with this value. Otherwise, fill by repeating the last element.
    F
        Returns:
            new_tensor (a torch tensor): output tensor with the dim size = `dim_size`.
    """
    assert tensor.shape[dim] > 0, "Input tensor must have at least 1 element"

    original_dim_size = tensor.shape[dim]
    # Trim
    if original_dim_size > dim_size:
        indices = torch.arange(dim_size)
        new_tensor = torch.index_select(tensor, dim, indices)
    # or Fill
    elif original_dim_size < dim_size:
        # fill with given value
        if fill_value is not None:
            fill_shape = list(tensor.shape)
            fill_shape[dim] = dim_size - original_dim_size
            fill = torch.full(fill_shape, fill_value)
        # or repeat last element
        else:
            shape = [1 for _ in range(tensor.ndim)]
            indices = torch.ones(shape).long()
            indices[0] = original_dim_size - 1
            last = torch.take_along_dim(tensor, indices, dim)

            repeat_shape = shape
            repeat_shape[dim] = dim_size - original_dim_size
            fill = last.repeat(repeat_shape)

        new_tensor = torch.cat([tensor, fill], dim=dim)
    else:
        new_tensor = tensor
    return new_tensor


def check_dicts_same_w_tensors(dict_1, dict_2, atol=1e-5) -> bool:
    """
    A utility function to check if two dicts are the same, with special handling of tensors.
    """
    if dict_1.keys() != dict_2.keys():
        return False

    for key in dict_1.keys():
        value1, value2 = dict_1[key], dict_2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.allclose(value1, value2, atol=atol):
                return False

        elif isinstance(value1, dict) and isinstance(value2, dict):
            # recursively check sub-dicts
            if not check_dicts_same_w_tensors(value1, value2, atol=atol):
                return False

        elif value1 != value2:
            return False

    return True


def concat_list_of_tensors(
    tensor_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate a list of tensors of (_, 3) into a single tensor of (N, 3), and returns both the stacked tensor and the first dims of the tensors in the list as a
    tensor of (len_of_list).
    """
    lengths_of_tensor = torch.tensor(
        [x.size(0) for x in tensor_list], dtype=torch.int64
    )

    if len(tensor_list) > 0:
        return torch.cat(tensor_list, dim=0), lengths_of_tensor
    else:
        return torch.tensor([]), torch.tensor([])


def unpack_list_of_tensors(
    stacked_tensor: torch.Tensor, lengths_of_tensors: torch.Tensor
) -> List[torch.Tensor]:
    """
    Unpack a stacked tensor of (N, 3) back to a list of tensors of (_, 3), according to each subtensor's lengths
    """
    assert lengths_of_tensors.sum().item() == stacked_tensor.size(
        0
    ), "The lengths_of_tensors do not sum to the length of the stacked tensor, {} vs {}".format(
        lengths_of_tensors.sum().item(), stacked_tensor.size(0)
    )

    tensor_list = []
    current_index = 0
    for length in lengths_of_tensors:
        tensor_list.append(stacked_tensor[current_index : current_index + length])
        current_index += length
    return tensor_list


def compute_bbox_corners_in_world(
    object_dimensions: torch.Tensor, Ts_world_object: torch.Tensor
) -> torch.Tensor:
    """
    Compute the 8 corners of the bounding box in world coordinates from object dimensions and T_world_object
    """
    num_obbs = object_dimensions.shape[0]

    corners_in_world_list = []
    # TODO: consider make this batched operation
    for i in range(num_obbs):
        # Extract object dimensions as a [8, 3] np array
        half_extents = object_dimensions[i] / 2.0
        hX = half_extents[0]
        hY = half_extents[1]
        hZ = half_extents[2]

        corners_in_object = np.array(
            [
                [-hX, -hY, -hZ],
                [hX, -hY, -hZ],
                [hX, hY, -hZ],
                [-hX, hY, -hZ],
                [-hX, -hY, hZ],
                [hX, -hY, hZ],
                [hX, hY, hZ],
                [-hX, hY, hZ],
            ],
            dtype=np.float32,
        )  # (8, 3)

        T_world_object = SE3.from_matrix3x4(Ts_world_object[i].numpy())

        corners_in_world = T_world_object @ (corners_in_object.T)
        corners_in_world_list.append(
            torch.tensor(corners_in_world.T, dtype=torch.float32)
        )  # (8, 3)

    return torch.stack(corners_in_world_list, dim=0)  # (num_obbs, 8, 3)


def filter_obbs_by_confidence(
    obb_dict: Dict, confidence_score: torch.Tensor, confidence_lower_threshold: float
) -> Dict:
    """
    A simple filter function to filter the predictions by confidence score (lower_threshold).
    The obb_dict follows the format of ATEK obb3_gt_processor convention
    """
    filtered_indices = confidence_score > confidence_lower_threshold
    # Make it 1-D if it's a scalar
    if filtered_indices.dim() == 0:
        filtered_indices = filtered_indices.unsqueeze(0)

    result_dict = {}

    for key, val in obb_dict.items():
        if isinstance(val, torch.Tensor):
            result_dict[key] = val[filtered_indices]
        if isinstance(val, List):
            result_dict[key] = [
                s for s, selected_flag in zip(val, filtered_indices) if selected_flag
            ]

    return result_dict


def filter_obbs_by_confidence_all_cams(
    all_cam_dict: Dict,
    confidence_score: torch.Tensor,
    confidence_lower_threshold: float,
) -> Dict:
    """
    A simple filter function to filter the predictions by confidence score (lower_threshold).
    The obb_dict follows the format of ATEK obb3_gt_processor convention
    """
    result_dict = {}

    for camera_label, single_cam_dict in all_cam_dict.items():
        result_dict[camera_label] = filter_obbs_by_confidence(
            obb_dict=single_cam_dict,
            confidence_score=confidence_score,
            confidence_lower_threshold=confidence_lower_threshold,
        )

    return result_dict

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

from typing import List, Union

import numpy as np
import seaborn as sns

import torch
from atek_v1.data_preprocess.data_schema import Frameset, FramesetGroup


def get_rate_stats(timestamps_ns: List[int]):
    deltas_ns = np.diff(timestamps_ns)
    deltas_ms = deltas_ns / 1000_000
    framerate_hz = 1 / np.mean(deltas_ms / 1000)
    mean_delta_ms = np.mean(deltas_ms)
    std_delta_ms = np.std(deltas_ms)
    max_delta_ms = np.max(deltas_ms)
    min_delta_ms = np.min(deltas_ms)
    return {
        "rate_hz": framerate_hz,
        "mean_delta_ms": mean_delta_ms,
        "std_delta_ms": std_delta_ms,
        "max_delta_ms": max_delta_ms,
        "min_delta_ms": min_delta_ms,
    }


def insert_and_check(my_dict, key, value):
    """
    Helper function to insert a new key-value pair into a dictionary,
    or check that the existing value matches the new one
    """
    if key not in my_dict:
        my_dict[key] = value
    elif my_dict[key] == value:
        pass
    else:
        raise AssertionError(f"Key {key} exists with a different value {my_dict[key]}")


def strict_update_dict(dict_A, dict_B):
    """
    Helper function to update a dictionary A with values from another dictionary B.
    The update will either insert a new key-value pair into a dictionary,
    or check that the existing value matches the new one
    """
    for k, v in dict_B.items():
        insert_and_check(dict_A, k, v)


def unify_list(input_list: List[int]):
    """
    Helper function to convert a list of integers to a list of unique values and their corresponding indices.
    """
    input_array = np.array(input_list)
    unique_values, indices = np.unique(input_array, return_index=True)

    unique_values = [int(val) for val in unique_values]
    indices = [int(idx) for idx in indices]
    return unique_values, indices


def check_all_same_member(values, member: str):
    # Extract the specified member from each object in the list
    member_values = [getattr(obj, member) for obj in values]
    # Check if all the extracted values are the same
    return len(set(member_values)) == 1


def generate_disjoint_colors(n):
    palette = sns.color_palette("husl", n)
    return [tuple(int(255 * x) for x in color) for color in palette]


def unify_object_target(
    unified_target: Union[FramesetGroup, Frameset],
):
    """
    Helper function to union the multiple frames object targets into
    the frameset level object target.
    """
    if isinstance(unified_target, FramesetGroup):
        member_name = "framesets"
    elif isinstance(unified_target, Frameset):
        member_name = "frames"
    else:
        raise ValueError("Invalid type for the unified target")

    unified_target.category_id_to_name = {}
    unified_target.object_instance_ids = []
    unified_target.object_category_ids = []
    unified_target.Ts_world_object = []
    unified_target.object_dimensions = []
    for member in getattr(unified_target, member_name):
        strict_update_dict(
            unified_target.category_id_to_name, member.category_id_to_name
        )
        unified_target.object_instance_ids += member.object_instance_ids
        unified_target.object_category_ids += member.object_category_ids
        unified_target.Ts_world_object += member.Ts_world_object
        unified_target.object_dimensions += member.object_dimensions

    assert (
        len(unified_target.object_instance_ids)
        == len(unified_target.object_category_ids)
        == len(unified_target.Ts_world_object)
        == len(unified_target.object_dimensions)
    ), (
        f"{len(unified_target.object_instance_ids)}, {len(unified_target.object_category_ids)}, "
        f"{len(unified_target.Ts_world_object)}, {len(unified_target.object_dimensions)}"
    )

    # Remove the duplicate object
    unified_target.object_instance_ids, unify_ids = unify_list(
        unified_target.object_instance_ids
    )
    unified_target.object_category_ids = [
        unified_target.object_category_ids[id] for id in unify_ids
    ]
    unified_target.Ts_world_object = [
        unified_target.Ts_world_object[id] for id in unify_ids
    ]
    unified_target.object_dimensions = [
        unified_target.object_dimensions[id] for id in unify_ids
    ]


def value_to_tensor(value: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    """
    Convert a numpy array or a list of same shape numpy array tensor to a tensor.
    """
    result = None
    if isinstance(value, np.ndarray):
        result = torch.from_numpy(value)
    elif isinstance(value, list):
        result = (
            torch.stack([torch.from_numpy(item) for item in value])
            if len(value) > 0
            else None
        )
    else:
        raise ValueError(f"Unsupported type {type(value)}")

    return result

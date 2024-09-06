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

import copy
import csv
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch
import yaml


def load_category_mapping_from_csv(
    category_mapping_csv_file: str,
) -> Dict:
    """
    Load the category mapping from a CSV file.

    Args:
        category_mapping_csv_file (str): The path to the category mapping CSV file.
        The CSV file should contain exactly 3 Columns, representing "old_category_name or prototype_name", "atek_category_name", "atek_category_id".

    Returns:
        Dict: The category mapping dictionary in the format of:
            {
                "old_cat/prototype_name"： [“cat_name”, category_id],
                ...
            }
    """
    with open(category_mapping_csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 3, "Expected 3 columns in the category mapping csv file"
        assert (
            header[1] == "ATEK Category Name" and header[2] == "ATEK Category Id"
        ), f"Column names must be  ATEK Category Name and ATEK Category Id, but got {header[1]} and {header[2]} instead."
        category_mapping = {rows[0]: (rows[1], rows[2]) for rows in reader}
    return category_mapping


def set_nested_dict_value(
    nested_dict: Dict,
    keys_as_path: List[str],
    value: Dict,
) -> None:
    """
    A helper function that sets a value in a nested dictionary according to the given "keys_as_path"

    :param nested_dict: The dictionary to be modified.
    :param path: A list of keys representing the path to the value in the nested dict.
    :param value: The value to be set.
    """
    current_dict = nested_dict
    for key in keys_as_path[:-1]:
        # create empty dict if not exist
        if key not in current_dict or not isinstance(current_dict[key], dict):
            current_dict[key] = {}
        current_dict = current_dict[key]

    current_dict[keys_as_path[-1]] = value


def separate_tensors_from_dict(gt_dict: Dict) -> Tuple[Dict, Dict]:
    """
    Extract tensors from a nested dict, return them in a flattened dict with new key being the concatenation of the keys' path, separated by `+`.
    E.g. "key1+key2+key3".
    Also return a copy of the original dict with the tensors removed.
    """
    tensor_dict = {}
    gt_dict_no_tensors = {}

    def recursive_extraction_helper(current_dict: Dict, keys_as_path: List[str]):
        for key, value in current_dict.items():
            # Recursively traverse nested dictionaries
            if isinstance(value, dict):
                recursive_extraction_helper(value, keys_as_path + [key])

            # Reached a leaf node, if it is a tensor, add it to tensor_dict
            elif isinstance(value, torch.Tensor):
                # Flatten the keys' path into a single string key
                flattened_tensor_key_name = "+".join(keys_as_path + [key])
                tensor_dict[flattened_tensor_key_name] = value
            else:
                # Preserve non-tensor values
                set_nested_dict_value(gt_dict_no_tensors, keys_as_path + [key], value)

    recursive_extraction_helper(gt_dict, [])

    return (gt_dict_no_tensors, tensor_dict)


def merge_tensors_into_dict(gt_dict_no_tensors: Dict, tensor_dict: Dict) -> Dict:
    """
    Essentially the reverse of `separate_tensors_from_dict`.
    """
    gt_dict = copy.deepcopy(gt_dict_no_tensors)

    for concat_keys, tensor_value in tensor_dict.items():
        keys_as_path = concat_keys.split("+")
        set_nested_dict_value(gt_dict, keys_as_path, tensor_value)

    return gt_dict


def load_yaml_and_extract_tar_list(yaml_path: str) -> List[str]:
    """
    Load a YAML file and extract URLs or convert relative paths to absolute paths
    from the specified number of sequences. If num_sequences is not specified or
    is larger than the available sequences, all sequences will be processed.
    Args:
    yaml_path (str): The path to the YAML file.
    num_sequences (Optional[int]): The number of sequences to extract data from. Defaults to None.
    Returns:
    List[str]: A list of URLs or absolute paths from the specified number of sequences.
    """
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)
    urls_or_paths = []
    yaml_dir = os.path.dirname(yaml_path)
    sequences = data.get("tars", {})
    for sequence_name, sequence_paths_or_urls in sequences.items():
        for single_sequence_path_or_url in sequence_paths_or_urls:
            if urlparse(single_sequence_path_or_url).scheme in ["http", "https"]:
                urls_or_paths.append(single_sequence_path_or_url)
            else:
                absolute_path = os.path.join(yaml_dir, single_sequence_path_or_url)
                urls_or_paths.append(absolute_path)
    return urls_or_paths

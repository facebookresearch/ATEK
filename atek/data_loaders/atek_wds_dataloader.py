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

import io
import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import torch
import webdataset as wds

from atek.util.file_io_utils import merge_tensors_into_dict
from atek.util.tensor_utils import unpack_list_of_tensors


def process_wds_sample(sample: Dict):
    """
    Process one webdataset sample.
    key_process_fn is an optional function helps handling key processing efficiently.
    it takes a key string and return a new processed key string or None after processing.
    None return means we will keep that key for processing.
    TODO: maybe add decryption as Callable
    """
    sample_as_dict = {}
    to_be_stacked_images = {}
    for k, v in sample.items():
        if k in ["__key__", "__local_path__", "__url__"]:
            sample_as_dict[k] = v
        else:
            key_wo_extension, extension_name = k.rsplit(".", 1)
            # Need to restack each image back into tensor of shape `[num_frame, C, H, W]`
            if extension_name == "jpeg":
                # Images are named as `${root_name}_{image_index}.jpeg` in WDS.
                # Break the name into root_name and image_index
                image_root_name = k.split("_")[0]
                image_index = int(k.split("_")[1].split(".")[0])

                if image_root_name not in to_be_stacked_images:
                    to_be_stacked_images[image_root_name] = {}

                # WDS-saved image are loaded as [3, H, W], where single-channel images are simply duplicated for 2 more channels.
                # convert to tensor of shape [F, C, H, W]
                if "rgb" not in image_root_name:
                    image = v[0:1, :, :]
                else:
                    image = v

                # append current image to the correct "stack"
                to_be_stacked_images[image_root_name][image_index] = image

            # Tensors
            elif extension_name == "pth":
                tensor_value = v
                if tensor_value.dtype == torch.float64:
                    tensor_value = tensor_value.float()
                sample_as_dict[key_wo_extension] = tensor_value
            # Dictionary
            elif extension_name == "json":
                sample_as_dict[key_wo_extension] = v
            # string values
            elif extension_name == "txt":
                sample_as_dict[key_wo_extension] = v
            else:
                raise ValueError(f"Unsupported file type in wds {k}")

    # restack images to tensor of [num_frames, C, H, W], but need to re-arrange image order first!
    for image_root_name, img_dict in to_be_stacked_images.items():
        # may have missing frames
        sorted_image_indices = sorted(img_dict.keys())
        image_list_with_correct_order = []

        for index in sorted_image_indices:
            image_list_with_correct_order.append(img_dict[index])

        # Convert each frame to the desired shape
        sample_as_dict[image_root_name] = torch.stack(
            image_list_with_correct_order, dim=0
        )

    # unpack semidense points from a stacked tensor back to List of tensors
    for key in ["points_world", "points_inv_dist_std", "points_dist_std"]:
        if f"msdpd#{key}+stacked" in sample_as_dict:
            sample_as_dict[f"msdpd#{key}"] = unpack_list_of_tensors(
                stacked_tensor=sample_as_dict[f"msdpd#{key}+stacked"],
                lengths_of_tensors=sample_as_dict[f"msdpd#points_world_lengths"],
            )
    # For tensors starting with "GtData#...", merge them back into GT dict
    keys_to_pop = []
    for key, value in sample_as_dict.items():
        if key.startswith("gt_data#") and isinstance(value, torch.Tensor):
            # merge into sample_dict["gt_data"]
            tensor_key_to_be_merged = key.replace("gt_data#", "")
            sample_as_dict["gt_data"] = merge_tensors_into_dict(
                sample_as_dict["gt_data"], {tensor_key_to_be_merged: value}
            )
            keys_to_pop.append(key)

    # pop the original gt tensors from sample_dict
    for tensor_key in keys_to_pop:
        sample_as_dict.pop(tensor_key)

    return sample_as_dict


def select_and_remap_dict_keys(
    sample_dict: Dict[str, Any], key_mapping: Dict[str, str]
) -> Dict[str, Any]:
    """
    Data transform function to modify the sample by selecting and remapping keys according to key_mapping
    """
    remapped_sample = {}
    for k, v in sample_dict.items():
        if k in ["__key__", "__url__"]:
            remapped_sample[k] = v
        elif k in key_mapping:
            new_key = key_mapping[k]
            remapped_sample[new_key] = v
    return remapped_sample


def atek_default_collation_fn(samples):
    """
    Take a collection of samples (dictionaries) from ATEK WDS, and collate them into a batch.
    :param samples: list of sample, each sample is a dict.
    """
    if len(samples) == 0:
        return {}

    assert isinstance(samples[0], dict)

    """
    TODO: add this feature back when needed
    # Get the common keys between samples. This is required when we train with
    # multiple datasets with samples having different keys.
    common_keys = set(samples[0].keys())
    for sample in samples[1:]:
        common_keys &= set(sample.keys())
    """

    batched_dict = {}
    for key in samples[0].keys():
        # first insert values of the same key into a list
        values_as_list = []
        for sample in samples:
            values_as_list.append(sample[key])

        # For tensor list, check if they can be stacked. If so, stack them.
        stackable_flag = False
        if isinstance(values_as_list[0], torch.Tensor):
            stackable_flag = True
            first_tensor_shape = values_as_list[0].shape
            first_tensor_dtype = values_as_list[0].dtype
            for tensor_i in values_as_list:
                if (
                    tensor_i.shape != first_tensor_shape
                    or tensor_i.dtype != first_tensor_dtype
                ):
                    print(
                        f"Tensors for {key} have different shapes or dtypes, cannot be stacked, keep as a list"
                    )
                    stackable_flag = False
                    break

        if stackable_flag:
            batched_dict[key] = torch.stack(values_as_list, dim=0)
        else:
            batched_dict[key] = values_as_list

    return batched_dict


def simple_list_collation_fn(batch):
    # Simply collate as a list
    return list(batch)


def load_atek_wds_dataset(
    urls: List[str],
    nodesplitter: Callable = wds.shardlists.single_node_only,
    dict_key_mapping: Optional[Dict[str, str]] = None,
    data_transform_fn: Optional[Callable] = None,
    batch_size: Optional[int] = None,
    collation_fn: Optional[Callable] = atek_default_collation_fn,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
) -> wds.WebDataset:
    # 1. load WDS samples back as dicts
    wds_dataset = (
        wds.WebDataset(urls, nodesplitter=nodesplitter)
        .decode(wds.imagehandler("torchrgb8"))
        .map(process_wds_sample)
    )

    # 2. random shuffle
    if shuffle_flag:
        wds_dataset = wds_dataset.shuffle(1000)

    # 3. remap dict keys
    if dict_key_mapping is not None:
        wds_dataset = wds_dataset.map(
            partial(select_and_remap_dict_keys, key_mapping=dict_key_mapping)
        )

    # 4. apply data transforms
    if data_transform_fn is not None:
        wds_dataset = wds_dataset.compose(data_transform_fn)

    # 5. batch samples
    if batch_size is not None:
        wds_dataset = wds_dataset.batched(batch_size, collation_fn=collation_fn)

    # 6. repeat dataset
    if repeat_flag:
        wds_dataset = wds_dataset.repeat()

    return wds_dataset


def create_native_atek_dataloader(
    urls: List[str],
    nodesplitter: Callable = wds.shardlists.single_node_only,
    dict_key_mapping: Optional[Dict[str, str]] = None,
    data_transform_fn: Optional[Callable] = None,
    collation_fn: Optional[Callable] = atek_default_collation_fn,
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:

    wds_dataset = load_atek_wds_dataset(
        urls=urls,
        nodesplitter=nodesplitter,
        dict_key_mapping=dict_key_mapping,
        data_transform_fn=data_transform_fn,
        batch_size=batch_size,
        collation_fn=collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )

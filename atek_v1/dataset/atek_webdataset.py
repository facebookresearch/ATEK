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

import json
from functools import partial

from typing import Callable, Dict, List, Optional

import torch
import webdataset as wds


def identity_key_process(key: str):
    """
    An identity key process function to keep the original keys.
    """
    return key


def convert_tensors_to_float32(obj):
    """
    Recursively converts all tensors in a dictionary, nested dictionary, or list
    from float64 to float32.
    """
    if isinstance(obj, dict):
        return {k: convert_tensors_to_float32(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_float32(item) for item in obj]
    elif isinstance(obj, torch.Tensor) and obj.dtype == torch.float64:
        return obj.float()  # Converts float64 tensor to float32
    else:
        return obj


def process_sample(sample: Dict):
    """
    Process one webdataset sample.
    key_process_fn is an optional function helps handling key processing efficiently.
    it takes a key string and return a new processed key string or None after processing.
    None return means we will keep that key for processing.
    """
    converted_sample = {}
    images = {}
    for k, v in sample.items():
        if k in ["__key__", "__url__"]:
            converted_sample[k] = v
        elif k.endswith(".jpeg"):
            image_key = k.split("_")[0]
            if image_key not in images:
                images[image_key] = []
            images[image_key].append(v)
        elif k.endswith(".pth"):
            assert isinstance(v, dict)
            converted_sample.update(v)
        elif k.endswith(".json"):
            assert isinstance(v, dict)
            converted_sample.update(v)
        else:
            raise ValueError(f"Unsupported file type in wds {k}")

    for k, v in images.items():
        converted_sample[k] = torch.stack(v)

    return convert_tensors_to_float32(converted_sample)


def select_sample_keys(sample, select_key_fn: Callable[[str], bool]):
    """
    Data transform function to modify the sample by poping not selected keys.
    """
    keys_to_pop = []
    for k in sample.keys():
        if k in ["__key__", "__url__"]:
            continue
        elif not select_key_fn(k):
            keys_to_pop.append(k)

    for k in keys_to_pop:
        sample.pop(k)
    return sample


def remap_sample_keys(sample, remap_key_fn: Callable[[str], str]):
    """
    Data transform function to remap the sample keys to user convenient keys.
    """
    remapped_sample = {}
    for k, v in sample.items():
        if k in ["__key__", "__url__"]:
            remapped_sample[k] = v
        else:
            new_key = remap_key_fn(k)
            remapped_sample[new_key] = v
    return remapped_sample


def atek_collation_fn(samples):
    """Take a collection of samples (dictionaries) and create a batch.
    :param dict samples: list of sample
    """
    assert isinstance(samples[0], dict)

    # Get the common keys between samples. This is required when we train with
    # multiple datasets with samples having different keys.
    common_keys = set(samples[0].keys())
    for sample in samples[1:]:
        common_keys &= set(sample.keys())

    batch = {}
    for k in common_keys:
        value = []
        for sample in samples:
            value.append(sample[k])
        if isinstance(value[0], torch.Tensor) and ("object" not in k):
            value = torch.stack(value)
        batch[k] = value

    return batch


def create_atek_webdataset(
    urls,
    batch_size: Optional[int] = None,
    collation_fn: Callable = atek_collation_fn,
    nodesplitter: Callable = wds.shardlists.single_node_only,
    select_key_fn: Optional[Callable[[str], bool]] = None,
    remap_key_fn: Optional[Callable[[str], str]] = None,
    data_transform_fn: Optional[Callable] = None,
    repeat: bool = False,
) -> wds.WebDataset:
    """
    A general standard webdataset processing data pipeline pattern:
    urls-> full atek dict-> [select keys] -> [remap keys] -> [more transform] -> [batch collation]
    Note that we keep the collation in the dataset part for better performance.
    A custom collation function may be required to be compatible with the custom data transform added.

    nodesplitter: set the node splitter properly to handle the multiple node/gpu training. Node that
    wds shardlists splitting function may not work as expected because of different distributed training
    settings. The safest way is to do the splitting manually or write the custom splitting function.
    repeat: set to true makes the dataset to be a infinite iterable dataset.
    """

    dataset = (
        wds.WebDataset(urls, nodesplitter=nodesplitter)
        .decode("torchrgb")
        .map(process_sample)
    )

    if select_key_fn is not None:
        dataset = dataset.map(partial(select_sample_keys, select_key_fn=select_key_fn))
    if remap_key_fn is not None:
        dataset = dataset.map(partial(remap_sample_keys, remap_key_fn=remap_key_fn))
    if data_transform_fn is not None:
        dataset = dataset.compose(data_transform_fn)
    if batch_size is not None:
        dataset = dataset.batched(batch_size, collation_fn=collation_fn)
    if repeat:
        dataset = dataset.repeat()

    return dataset


def create_wds_dataloader(
    atek_dataset: wds.WebDataset,
    num_workers=4,
    pin_memory=False,
):
    """
    Note that the batch size is more efficient to set during creating the webdataset.
    """
    return torch.utils.data.DataLoader(
        atek_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

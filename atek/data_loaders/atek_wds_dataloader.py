# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import io
import json
from functools import partial
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import torch
import webdataset as wds

from atek.data_preprocess.util.file_io_utils import unpack_list_of_tensors
from torchvision.io import read_image


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
        if k in ["__key__", "__url__"]:
            sample_as_dict[k] = v
        else:
            key_wo_extension, extension_name = k.rsplit(".", 1)
            # Images are named as `${root_name}_{frame_id}.jpeg` in WDS.
            # Need to restack each image back into tensor of shape `[num_frame, C, H, W]`
            if extension_name == "jpeg":
                image_root_name = k.split("_")[0]

                if image_root_name not in to_be_stacked_images:
                    to_be_stacked_images[image_root_name] = []

                # WDS-saved image are loaded as [3, H, W], where single-channel images are simply duplicated for 2 more channels.
                # convert to tensor of shape [F, C, H, W]
                if "rgb" not in image_root_name:
                    image = v[0:1, :, :]
                else:
                    image = v

                # append current image to the correct "stack"
                to_be_stacked_images[image_root_name].append(image)

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

    # restack images to tensor of [num_frames, C, H, W]
    for image_root_name, image_frames in to_be_stacked_images.items():
        # Convert each frame to the desired shape
        sample_as_dict[image_root_name] = torch.stack(image_frames, dim=0)

    # unpack semidense points from a stacked tensor back to List of tensors
    sample_as_dict["msdpd#points_world"] = unpack_list_of_tensors(
        stacked_tensor=sample_as_dict["msdpd#stacked_points_world"],
        lengths_of_tensors=sample_as_dict["msdpd#points_world_lengths"],
    )
    sample_as_dict["msdpd#points_inv_dist_std"] = unpack_list_of_tensors(
        stacked_tensor=sample_as_dict["msdpd#stacked_points_inv_dist_std"],
        lengths_of_tensors=sample_as_dict["msdpd#points_world_lengths"],
    )

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


def load_atek_wds_dataset(
    urls: List[str],
    dict_key_mapping: Optional[Dict[str, str]] = None,
    data_transform_fn: Optional[Callable] = None,
) -> wds.WebDataset:

    # first, load WDS samples back as dicts
    wds_dataset = (
        wds.WebDataset(urls)
        .decode(wds.imagehandler("torchrgb8"))
        .map(process_wds_sample)
    )

    # second, remap dict keys
    if dict_key_mapping is not None:
        wds_dataset = wds_dataset.map(
            partial(select_and_remap_dict_keys, key_mapping=dict_key_mapping)
        )

    # third, apply data transforms
    if data_transform_fn is not None:
        wds_dataset = wds_dataset.compose(data_transform_fn)

    return wds_dataset

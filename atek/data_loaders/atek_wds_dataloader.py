# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import io
import json
from typing import Dict, List

import numpy as np

import torch
import webdataset as wds
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
                # sample_as_dict[key_wo_extension] = json.loads(v.decode("utf-8"))
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

    return sample_as_dict


def load_atek_wds_dataset(
    urls: List[str],
) -> wds.WebDataset:
    wds_dataset = (
        wds.WebDataset(urls)
        .decode(wds.imagehandler("torchrgb8"))
        .map(process_wds_sample)
    )

    return wds_dataset
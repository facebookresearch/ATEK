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
import os
import re
from dataclasses import dataclass

from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import webdataset as wds

from atek_v1.data_preprocess.data_schema import Frame, Frameset, FramesetGroup
from PIL import Image


def tensorize_value(x):
    """
    Helper function to recursively tensorize the input object. Only expect list nested input.
    This function will convert the value to a single tensor.
    Return None for empty list.
    """
    if isinstance(x, torch.Tensor):
        return x
    elif isinstance(x, list):
        if len(x) == 0:
            return None
        return torch.stack([tensorize_value(xi) for xi in x], dim=0)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise ValueError(f"Unsupported type {type(x)}")


def get_info_json(
    flatten_fsg_dict: Dict,
    key_pattern: re.Pattern,
    selected_fields: List,
):
    """
    A helper function to generate a json string from selected dict values based on keys.
    This is better suitable for metadata.
    """
    info_json_dict = {}
    for k, v in flatten_fsg_dict.items():
        if key_pattern.search(k) and k.split("+")[-1] in selected_fields:
            info_json_dict[k] = v
    return info_json_dict


def get_tensor_dict(
    flatten_fsg_dict: Dict,
    key_pattern: re.Pattern,
    selected_fields: List,
    return_as_list_of_tensor: bool = False,
):
    """
    A helper function to generate a dict tensor from selected dict values based on keys.
    This is better suitable for tensor style of data.

    return_as_list_of_tensor: by default set to false so that try the best to convert values
    into a single tensor. However, some data is not aligned in length for example objects, so we
    return a list of tensors instead by setting return_as_list_of_tensor to true.
    """
    tensor_dict = {}
    for k, v in flatten_fsg_dict.items():
        if key_pattern.search(k) and k.split("+")[-1] in selected_fields:
            if return_as_list_of_tensor:
                tensor_dict[k] = [tensorize_value(vi) for vi in v]
            else:
                tensor_dict[k] = tensorize_value(v)
    return tensor_dict


def add_frame_images_to_wds(flatten_fsg_dict, wds_dict):
    frame_key_pattern = Frame.flatten_dict_key_pattern()
    for k, v in flatten_fsg_dict.items():
        if frame_key_pattern.search(k) and k.split("+")[-1] in Frame.image_fields():
            for i, image in enumerate(v):
                wds_dict[f"{k}_{i}.jpeg"] = image


def add_frame_camera_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_camera_info.json"] = get_info_json(
        flatten_fsg_dict, Frame.flatten_dict_key_pattern(), Frame.camera_info_fields()
    )


def add_frame_camera_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_camera_data.pth"] = get_tensor_dict(
        flatten_fsg_dict, Frame.flatten_dict_key_pattern(), Frame.camera_data_fields()
    )


def add_frame_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_info.json"] = get_info_json(
        flatten_fsg_dict, Frame.flatten_dict_key_pattern(), Frame.input_info_fields()
    )


def add_frame_trajectory_data_to_wds(flatten_dict, wds_dict):
    wds_dict["frame_trajectory_data.pth"] = get_tensor_dict(
        flatten_dict,
        Frame.flatten_dict_key_pattern(),
        Frame.trajectory_data_fields(),
    )


def add_frame_object_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_object_info.json"] = get_info_json(
        flatten_fsg_dict,
        Frame.flatten_dict_key_pattern(),
        Frame.object_info_fields(),
    )


def add_frame_object_2d_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_object_2d.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        Frame.flatten_dict_key_pattern(),
        Frame.object_2d_data_fields(),
        return_as_list_of_tensor=True,
    )


def add_frame_object_3d_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frame_object_3d.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        Frame.flatten_dict_key_pattern(),
        Frame.object_3d_data_fields(),
        return_as_list_of_tensor=True,
    )


def add_frameset_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_info.json"] = get_info_json(
        flatten_fsg_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.input_info_fields(),
    )


def add_frameset_camera_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_camera_data.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.camera_data_fields(),
    )


def add_frameset_trajectory_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_trajectory_data.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.trajectory_data_fields(),
    )


def add_frameset_semidense_points_to_wds(flatten_dict, wds_dict):
    wds_dict["frameset_semidense_points_data.pth"] = get_tensor_dict(
        flatten_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.semidense_points_fields(),
        return_as_list_of_tensor=True,
    )


def add_frameset_object_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_object_info.json"] = get_info_json(
        flatten_fsg_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.object_info_fields(),
    )


def add_frameset_object_3d_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_object_3d.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        Frameset.flatten_dict_key_pattern(),
        Frameset.object_3d_data_fields(),
        return_as_list_of_tensor=True,
    )


def add_frameset_group_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_group_info.json"] = get_info_json(
        flatten_fsg_dict,
        FramesetGroup.flatten_dict_key_pattern(),
        FramesetGroup.input_info_fields(),
    )


def add_frameset_group_trajectory_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_group_trajectory_data.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        FramesetGroup.flatten_dict_key_pattern(),
        FramesetGroup.trajectory_data_fields(),
    )


def add_frameset_group_object_info_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_group_object_info.json"] = get_info_json(
        flatten_fsg_dict,
        FramesetGroup.flatten_dict_key_pattern(),
        FramesetGroup.object_info_fields(),
    )


def add_frameset_group_object_3d_data_to_wds(flatten_fsg_dict, wds_dict):
    wds_dict["frameset_group_object_3d.pth"] = get_tensor_dict(
        flatten_fsg_dict,
        FramesetGroup.flatten_dict_key_pattern(),
        FramesetGroup.object_3d_data_fields(),
    )


def encrypt_wds_dict(wds_dict, encryption_handler: Callable[[bytes], bytes]):
    """
    Encodes the given WDS dictionary based on the keys' file extension to different
    files. The dictionary is expected to have keys that end with .jpeg, .json, or .pth,
    and the corresponding values should be in a format that can be serialized into
    these formats.
    """
    encoded_dict = {}
    for k, v in wds_dict.items():
        if k == "__key__":
            encoded_dict[k] = v
        elif k.endswith(".jpeg"):
            # Assuming 'v' is a numpy array representing an image
            # Convert to PIL Image and then to bytes
            pil_image = Image.fromarray(v)
            with io.BytesIO() as output_bytes:
                pil_image.save(output_bytes, format="JPEG")
                encrypted_val = encryption_handler(output_bytes.getvalue())
            encoded_dict[k] = encrypted_val
        elif k.endswith(".json"):
            # Assuming 'v' is serializable to JSON
            encoded_dict[k] = encryption_handler(json.dumps(v).encode("utf-8"))
        elif k.endswith(".pth"):
            # Assuming 'v' is a PyTorch tensor or a dictionary of tensors
            with io.BytesIO() as buffer:
                torch.save(v, buffer)
                buffer.seek(0)
                encrypted_val = encryption_handler(buffer.read())
            encoded_dict[k] = encrypted_val
        else:
            raise ValueError(f"Unknown key extension {k}")

    return encoded_dict


@dataclass
class DataSelectionSettings:
    # Flag to include per-frame trajectory information.
    require_traj_for_frame: bool = False

    # Flag to include per-frame 2D object bounding boxes.
    require_obb2d_gt_for_frame: bool = False

    # Flag to include per-frame 3D object bounding boxes.
    require_obb3d_gt_for_frame: bool = False

    # Flag to include per-frameset trajectory information.
    require_traj_for_frameset: bool = False

    # Flag to include per-frameset semidense point cloud information
    require_semidense_points_for_frameset: bool = False

    # Flag to include per-frameset 3D object bounding boxes.
    require_obb3d_gt_for_frameset: bool = False

    # Flag to include overall frameset group trajectory information.
    require_traj_for_frameset_group: bool = False

    # Flag to include overall frameset group 3D object bounding boxes.
    require_obb3d_gt_for_frameset_group: bool = False


def convert_frameset_group_to_wds_dict(
    wds_id: int,
    frameset_group: FramesetGroup,
    wds_data_prefix_string: str,
    data_selection_settings: DataSelectionSettings,
):
    """
    Prepares a FramesetGroup object for serialization with WebDataset by flattening
    its structure and collating required fields into appropriate formats. The output is
    organized into multiple files, such as images, metadata JSONs, and tensor .pth files.

    Primitive types like ints and strs are stored in lists or dicts for clarity and
    readability as metadata. Numerical data like poses and bounding boxes are tensorized,
    while images are kept as NumPy arrays to facilitate easy image IO operations.

    The function selectively includes various pieces of trajectory and bounding box
    information based on specified requirements. This tailored serialization supports
    flexibility and efficiency in dataset storage and retrieval.

    Args:
        wds_id: Unique ID for the WDS sample.
        frameset_group: The FramesetGroup object to serialize.
        data_selection_settings: Settings for selecting data from the FramesetGroup.
        wds_data_prefix_string: Prefix string to prepend to `__key__`, which should be unique across datasets because WDS has assumptions that filenames across different tars need to be unique
    """
    flatten_fsg_dict = frameset_group.to_flatten_dict()

    wds_dict = {"__key__": f"{wds_data_prefix_string}_fsg_{wds_id:06}"}

    # Default fields which we want to keep always.
    add_frame_info_to_wds(flatten_fsg_dict, wds_dict)
    add_frame_images_to_wds(flatten_fsg_dict, wds_dict)
    add_frame_camera_info_to_wds(flatten_fsg_dict, wds_dict)
    add_frame_camera_data_to_wds(flatten_fsg_dict, wds_dict)
    add_frameset_info_to_wds(flatten_fsg_dict, wds_dict)
    add_frameset_camera_data_to_wds(flatten_fsg_dict, wds_dict)
    add_frameset_group_info_to_wds(flatten_fsg_dict, wds_dict)

    # Trajectory data.
    if data_selection_settings.require_traj_for_frame:
        add_frame_trajectory_data_to_wds(flatten_fsg_dict, wds_dict)
    if data_selection_settings.require_traj_for_frameset:
        add_frameset_trajectory_data_to_wds(flatten_fsg_dict, wds_dict)
    if data_selection_settings.require_traj_for_frameset_group:
        add_frameset_group_trajectory_data_to_wds(flatten_fsg_dict, wds_dict)

    # Semidense point cloud data.
    if data_selection_settings.require_semidense_points_for_frameset:
        add_frameset_semidense_points_to_wds(flatten_fsg_dict, wds_dict)

    # Object data
    if (
        data_selection_settings.require_obb2d_gt_for_frame
        or data_selection_settings.require_obb3d_gt_for_frame
    ):
        add_frame_object_info_to_wds(flatten_fsg_dict, wds_dict)
        if data_selection_settings.require_obb2d_gt_for_frame:
            add_frame_object_2d_data_to_wds(flatten_fsg_dict, wds_dict)
        if data_selection_settings.require_obb3d_gt_for_frame:
            add_frame_object_3d_data_to_wds(flatten_fsg_dict, wds_dict)
    if data_selection_settings.require_obb3d_gt_for_frameset:
        add_frameset_object_info_to_wds(flatten_fsg_dict, wds_dict)
        add_frameset_object_3d_data_to_wds(flatten_fsg_dict, wds_dict)
    if data_selection_settings.require_obb3d_gt_for_frameset_group:
        add_frameset_group_object_info_to_wds(flatten_fsg_dict, wds_dict)
        add_frameset_group_object_3d_data_to_wds(flatten_fsg_dict, wds_dict)

    return wds_dict


class AtekWdsWriter:
    """
    An interface to write data to a WebDataset (WDS) directory.
    """

    def __init__(
        self,
        output_path: str,
        prefix_string: str,
        data_selection_settings: DataSelectionSettings,
        max_samples_per_shard: int = 32,
        encryption_handler: Optional[Callable[[bytes], bytes]] = None,
    ):
        """
        Initialize the AtekWdsWriter object.

        Args:
            output_path (str): The path to the output directory where the WDS files will be written.
            prefix_string (str): A prefix string to prepend to the WDS file names.
            data_selection_settings (DataSelectionSettings): Settings for selecting data from the FramesetGroup.
            max_samples_per_shard (int, optional): The maximum number of samples to write to each WDS shard. Defaults to 32.
            encryption_handler (Optional[Callable[[bytes], bytes]], optional): A function to encrypt the WDS data. Defaults to None.
        """
        self.output_path = output_path
        self.prefix_string = prefix_string
        self.settings = data_selection_settings
        self.max_samples_per_shard = max_samples_per_shard
        self.encryption_handler = encryption_handler

        # Initialize sink when writing the first sample
        self.sink = None
        self.next_idx = 0

    def add_sample(self, frameset_group: FramesetGroup):
        """
        Add a sample to the WDS writer.

        Args:
            frameset_group (FramesetGroup): The FramesetGroup object to add to the WDS.
        """
        sample_dict = convert_frameset_group_to_wds_dict(
            self.next_idx, frameset_group, self.prefix_string, self.settings
        )
        if self.encryption_handler is not None:
            sample_dict = encrypt_wds_dict(sample_dict, self.encryption_handler)

        if self.sink is None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            self.sink = wds.ShardWriter(
                f"{self.output_path}/shards-%04d.tar",
                maxcount=self.max_samples_per_shard,
            )

        self.sink.write(sample_dict)
        self.next_idx += 1

    def close(self):
        """
        Close the WDS writer and flush any remaining data to disk.
        """
        if self.sink is not None:
            self.sink.close()

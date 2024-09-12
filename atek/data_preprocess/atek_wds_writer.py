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
import logging
import os

from typing import Dict, List, Tuple

import torch
import webdataset as wds

from atek.data_preprocess.atek_data_sample import AtekDataSample
from atek.util.file_io_utils import separate_tensors_from_dict
from atek.util.tensor_utils import concat_list_of_tensors
from omegaconf import DictConfig

SEMIDENSE_POINTS_FIELDS = [
    "msdpd#points_world",
    "msdpd#points_dist_std",
    "msdpd#points_inv_dist_std",
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_atek_sample_dict_to_wds_dict(
    index: int,
    atek_sample_dict: Dict,
    prefix_string: str,
) -> Dict:

    wds_dict = {"__key__": f"{prefix_string}_AtekDataSample_{index:06}"}

    for atek_key, atek_value in atek_sample_dict.items():
        # Semidense point data needs special handling later
        if atek_key in SEMIDENSE_POINTS_FIELDS:
            continue

        # GT needs special handling. The  tensors in the dict needs to be serialized as `.pth` file, while the rest stays in the dict.
        elif atek_key == "gt_data":
            gt_dict_no_tensor, tensor_dict = separate_tensors_from_dict(atek_value)
            wds_dict["gt_data.json"] = gt_dict_no_tensor
            for tensor_key, tensor_value in tensor_dict.items():
                wds_dict[f"gt_data#{tensor_key}.pth"] = tensor_value
            continue

        # Depth images should be directly saved as tensors
        elif atek_key.endswith("depth+images"):
            wds_dict[f"{atek_key}.pth"] = atek_value

        # Images needs to be separated into per-frame jpeg files
        elif atek_key.endswith("images"):
            assert isinstance(atek_value, torch.Tensor)
            # Transpose dimensions, [Frame, C, H, W] -> [Frame, H, W, C]
            image_frames_in_np = atek_value.numpy().transpose(0, 2, 3, 1)
            for id, img in enumerate(image_frames_in_np):
                new_key = f"{atek_key}_{id}.jpeg"
                wds_dict[new_key] = img if img.shape[-1] == 3 else img.squeeze()
            continue

        # For other fields, simply adds proper file extension to the key
        elif isinstance(atek_value, torch.Tensor):
            wds_dict[f"{atek_key}.pth"] = atek_value
        elif isinstance(atek_value, str):
            wds_dict[f"{atek_key}.txt"] = atek_value
        elif isinstance(atek_value, dict):
            wds_dict[f"{atek_key}.json"] = atek_value
        else:
            raise ValueError(f"Unsupported type {type(atek_value)} in ATEK WDS writer.")

    # Semidense point data needs to be flattended from List[Tensor (N, 3)] to a Tensor (M, 3) in order to be writable to WDS.
    # Therefore we store a `stacked_points_world` (Tensor [M, 3]) along with `points_world_lengths` (Tensor [num_frames]),
    # in order to unpack the stacked tensor later. Same for `points_inv_dist_std`.
    # obtain the "lengths" of each tensor in list.
    len_tensors = None
    for semidense_key in SEMIDENSE_POINTS_FIELDS:
        if semidense_key in atek_sample_dict:
            concatenated_tensor, current_len_tensors = concat_list_of_tensors(
                atek_sample_dict[semidense_key]
            )
            wds_dict[f"{semidense_key}+stacked.pth"] = concatenated_tensor
            if len_tensors is None:
                len_tensors = current_len_tensors.clone()
            else:
                assert torch.allclose(
                    len_tensors, current_len_tensors, atol=1
                ), f"The lengths for all semidense points data types should be the same! Instead got\n {current_len_tensors} in {semidense_key} vs \n {len_tensors}"
    if len_tensors is not None:
        wds_dict["msdpd#points_world_lengths.pth"] = len_tensors

    return wds_dict


class AtekWdsWriter:
    def __init__(self, output_path: str, conf: DictConfig) -> None:
        """
        INIT_DOC_STRING
        """
        self.output_path = output_path
        self.prefix_string = conf.prefix_string
        self.sink = None
        self.current_sample_idx = 0
        self.max_samples_per_shard = conf.max_samples_per_shard
        self.samples_in_current_shard = 0

        # A flag to indicate to remove last tar file, if it is not full. Default is False.
        self.remove_last_tar_if_not_full = (
            conf.remove_last_tar_if_not_full
            if "remove_last_tar_if_not_full" in conf
            else False
        )

    def add_sample(self, data_sample: AtekDataSample):
        """
        Add a sample to the WDS writer.
        """
        atek_sample_dict = data_sample.to_flatten_dict()
        wds_dict = convert_atek_sample_dict_to_wds_dict(
            self.current_sample_idx,
            atek_sample_dict=atek_sample_dict,
            prefix_string=self.prefix_string,
        )

        if self.sink is None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            self.sink = wds.ShardWriter(
                f"{self.output_path}/shards-%04d.tar",
                maxcount=self.max_samples_per_shard,
            )

        self.sink.write(wds_dict)
        self.current_sample_idx += 1
        self.samples_in_current_shard += 1

    def get_num_samples(self):
        return self.current_sample_idx

    def close(self):
        """
        Close the WDS writer and flush any remaining data to disk.
        """
        if self.sink is not None:
            self.sink.close()

        # Remove the last tar file if it has less than max_samples_per_shard samples
        if self.remove_last_tar_if_not_full:
            tar_files = [f for f in os.listdir(self.output_path) if f.endswith(".tar")]
            if (
                len(tar_files) > 0
                and self.current_sample_idx % self.max_samples_per_shard != 0
            ):
                last_tar_file = os.path.join(self.output_path, tar_files[-1])
                os.remove(last_tar_file)
            else:
                logger.warning("No tar files found in the output path.")

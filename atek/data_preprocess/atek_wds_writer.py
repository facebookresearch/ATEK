# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import copy
import os

from typing import Dict, List, Tuple

import torch
import webdataset as wds

from atek.data_preprocess.atek_data_sample import AtekDataSample
from atek.util.file_io_utils import separate_tensors_from_dict
from omegaconf import DictConfig


def convert_atek_data_sample_to_wds_dict(
    index: int,
    data_sample: AtekDataSample,
    prefix_string: str,
) -> Dict:
    flatten_dict = data_sample.to_flatten_dict()

    # GT dict needs some special handling. The tensors in the dict needs to be serialized as `.pth` file, while the rest stays in the dcit.
    if "gtdata.json" in flatten_dict:
        gt_dict_no_tensor, tensor_dict = separate_tensors_from_dict(
            flatten_dict["gtdata.json"]
        )
        flatten_dict["gtdata.json"] = gt_dict_no_tensor
        for tensor_key, tensor_value in tensor_dict.items():
            flatten_dict[f"gtdata#{tensor_key}.pth"] = tensor_value

    wds_dict = {"__key__": f"{prefix_string}_AtekDataSample_{index:06}"}
    wds_dict.update(flatten_dict)

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

    def add_sample(self, data_sample: AtekDataSample):
        """
        Add a sample to the WDS writer.
        """
        sample_dict = convert_atek_data_sample_to_wds_dict(
            self.current_sample_idx, data_sample, self.prefix_string
        )

        if self.sink is None:
            if not os.path.exists(self.output_path):
                os.makedirs(self.output_path)

            self.sink = wds.ShardWriter(
                f"{self.output_path}/shards-%04d.tar",
                maxcount=self.max_samples_per_shard,
            )

        self.sink.write(sample_dict)
        self.current_sample_idx += 1

    def get_num_samples(self):
        return self.current_sample_idx

    def close(self):
        """
        Close the WDS writer and flush any remaining data to disk.
        """
        if self.sink is not None:
            self.sink.close()

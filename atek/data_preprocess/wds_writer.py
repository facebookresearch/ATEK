# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os

from typing import Dict

import torch
import webdataset as wds

from atek.data_preprocess.atek_data_sample import AtekDataSample
from omegaconf import DictConfig


def serialize_tensors(obj):
    """
    Helper function to recursively serialize tensors in the input object. Only expect list, dict, or values as input
    This function will convert the value to lists.
    Return.
    """
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: serialize_tensors(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [serialize_tensors(value) for value in obj]
    else:
        return obj


def convert_atek_data_sample_to_wds_dict(
    index: int,
    data_sample: AtekDataSample,
    prefix_string: str,
) -> Dict:
    flatten_dict = data_sample.to_flatten_dict()

    # GT dict needs to be Json serializable
    if "GtData.json" in flatten_dict:
        flatten_dict["GtData.json"] = serialize_tensors(flatten_dict["GtData.json"])

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

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

# pyre-strict

import logging
import os

from typing import Any, Callable, Dict, Optional

from atek.data_loaders.atek_wds_dataloader import (
    process_wds_sample,
    select_and_remap_dict_keys,
)
from atek.data_loaders.cubercnn_model_adaptor import CubeRCNNModelAdaptor
from atek.data_preprocess.atek_data_sample import AtekDataSample

from atek.data_preprocess.sample_builders.obb_sample_builder import ObbSampleBuilder
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)
from omegaconf.omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AtekRawDataloaderAsCubercnn:
    def __init__(
        self, vrs_file: str, mps_files: Dict[str, str], conf: DictConfig
    ) -> None:
        # initialize the sample builder
        self.sample_builder = ObbSampleBuilder(
            conf=conf.processors, vrs_file=vrs_file, mps_files=mps_files, gt_files={}
        )

        # Create a subsampler
        self.subsampler = CameraTemporalSubsampler(
            vrs_file=vrs_file,
            conf=conf.camera_temporal_subsampler,
        )

        # Create a CubeRCNN model adaptor
        self.model_adaptor = CubeRCNNModelAdaptor()

    def __len__(self):
        return self.subsampler.get_total_num_samples()

    def get_timestamps_by_sample_index(self, index: int):
        return self.subsampler.get_timestamps_by_sample_index(index)

    def get_atek_sample_at_timestamp_ns(
        self, timestamp_ns: int
    ) -> Optional[AtekDataSample]:
        return self.sample_builder.get_sample_by_timestamp_ns(timestamp_ns)

    def get_model_specific_sample_at_timestamp_ns(
        self, timestamp_ns: int
    ) -> Optional[Dict]:
        atek_sample = self.sample_builder.get_sample_by_timestamp_ns(timestamp_ns)
        if atek_sample is None:
            logger.warning(
                f"Cannot retrieve valid atek sample at timestamp {timestamp_ns}"
            )
            return None

        # Flatten to dict
        atek_sample_dict = atek_sample.to_flatten_dict()

        # key remapping
        remapped_data_dict = select_and_remap_dict_keys(
            sample_dict=atek_sample_dict,
            key_mapping=self.model_adaptor.get_dict_key_mapping_all(),
        )

        # transform
        model_specific_sample_gen = self.model_adaptor.atek_to_cubercnn(
            [remapped_data_dict]
        )

        # Obtain a dict from a generator object
        model_specific_sample = next(model_specific_sample_gen)

        return model_specific_sample

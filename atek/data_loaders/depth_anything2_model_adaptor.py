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


from typing import Dict, List

from atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
    simple_list_collation_fn,
)

from webdataset.filters import pipelinefilter


class DepthAnything2ModelAdaptor:
    """
    A simple model adaptor class to convert ATEK WDS data to Depth Anything2 format, essentially just keep the RGB image for inference.
    Currently used for inference examples.
    """

    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {"mfcd#camera-rgb+images": "image"}
        return dict_key_mapping

    def atek_to_depth_anything2(self, data):
        for atek_wds_sample in data:
            sample = {}
            # Add images
            # from [1, C, H, W] to [H, W, C]
            image_torch = atek_wds_sample["image"].clone().detach()
            image_np = image_torch.squeeze(0).permute(1, 2, 0).numpy()
            sample["image"] = image_np
            yield sample


def load_atek_wds_dataset_as_depth_anything2(
    urls: List,
    batch_size: int,
    repeat_flag: bool,
    shuffle_flag: bool = False,
):
    adaptor = DepthAnything2ModelAdaptor()

    return load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=DepthAnything2ModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(adaptor.atek_to_depth_anything2)(),
        collation_fn=simple_list_collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

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

from argparse import Namespace
from typing import Dict

from atek_v1.dataset.omni3d_adapter import (
    create_omni3d_raw_dataset,
    create_omni3d_webdataset,
    ObjectDetectionMode,
)

from torch.utils.data import Dataset


def create_inference_dataset(
    data_path: str, args: Namespace, model_config: Dict
) -> Dataset:
    """
    Create dataset for inference pipeline.

    Args:
        data_path (str): path to a sequence
        args (Namespace): args with options related to dataset creation
        model_config (Dict): model config with options related to dataset creation

    Return:
        dataset (Dataset): a dataset instance for the inference pipeline
    """

    if args.model_name == "cubercnn":
        if args.data_type == "wds":
            dataset = create_omni3d_webdataset(
                [data_path],
                batch_size=1,
                repeat=False,
                category_id_remapping_json=model_config["cfg"].ID_MAP_JSON,
                object_detection_mode=ObjectDetectionMode[
                    model_config["cfg"].DATASETS.OBJECT_DETECTION_MODE
                ],
            )
        elif args.data_type == "raw":
            dataset = create_omni3d_raw_dataset(
                data_path,
                selected_device_number=0,
                rotate_image_cw90deg=args.rotate_image_cw90deg,
                target_image_resolution=(args.width, args.height),
            )
        else:
            raise ValueError(
                f"Unknown input data type for building dataset: {args.data_type}"
            )
    else:
        raise ValueError(f"Unknown model name for building dataset: {args.model_name}")

    return dataset

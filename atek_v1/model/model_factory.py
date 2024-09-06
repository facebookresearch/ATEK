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

from atek_v1.model.cubercnn import create_cubercnn_config, create_cubercnn_model


def create_inference_model(args: Namespace):
    """
    Create the model for inference pipeline, with the model config.

    Args:
        args (Namespace):

    Returns:
        model_config: dict-based detailed model configurations
        model: model used for the inference pipeline
    """
    if args.model_name == "cubercnn":
        model_config = create_cubercnn_config(args)
        model = create_cubercnn_model(model_config)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_name}")

    return model_config, model

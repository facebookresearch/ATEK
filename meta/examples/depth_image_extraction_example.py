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

import logging
import os

from atek.data_preprocess.processors.aria_camera_processor import identity_transform

from atek.data_preprocess.processors.depth_image_processor import DepthImageProcessor
from atek.data_preprocess.subsampling_lib.temporal_subsampler import (
    CameraTemporalSubsampler,
)

from omegaconf import OmegaConf

example_adt_data_dir = "/home/louy/Calibration_data_link/Atek/2024_05_07_EfmDataTest/adt_data_example/Apartment_release_clean_seq134/1WM103600M1292_optitrack_release_clean_seq134"
adt_config_path = os.path.join(
    "/home/louy/Calibration_data_link/Atek/2024_05_28_CubeRcnnTest/cubercnn_preprocess_adt_config.yaml"
)
config_file_path = os.path.join(
    "/home/louy/Calibration_data_link/Atek/2024_07_01_DepthExample/depth_example_config.yaml"
)

conf = OmegaConf.load(config_file_path)

# Construct DepthImageProcessor
depth_conf = conf.processors.rgb_depth
rgb_depth_processor = DepthImageProcessor(
    depth_vrs=os.path.join(example_adt_data_dir, "depth_images.vrs"),
    image_transform=identity_transform,  # if no transform is needed
    conf=depth_conf,
)

# Obtain depth images according to RGB images
subsampler = CameraTemporalSubsampler(
    vrs_file=os.path.join(
        example_adt_data_dir, "video.vrs"
    ),  # ADT depth and Aria are 2 separate VRS files
    conf=conf.camera_temporal_subsampler,
)

#
for i in range(subsampler.get_total_num_samples()):
    timestamps_ns = subsampler.get_timestamps_by_sample_index(i)
    depth_data = rgb_depth_processor.get_depth_data_by_timestamps_ns(
        timestamps_ns=timestamps_ns
    )
    if depth_data is None:
        print(f"--- warning: no depth data for timestamps {timestamps_ns}")
    else:
        depth_images = depth_data.images
        print(
            f"--- depth images for timestamps {timestamps_ns} are stored as tensor with shape {depth_images.shape}"
        )

    # Do what you need with the depth data. We recommend look at ATEK's `SampleBuilder` class to group these with other data and write to WDS files.

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

import faulthandler

import logging
import os
from logging import StreamHandler

from atek.data_preprocess.genera_atek_preprocessor_factory import (
    create_general_atek_preprocessor_from_conf,
)

from atek.data_preprocess.general_atek_preprocessor import GeneralAtekPreprocessor
from omegaconf import OmegaConf

faulthandler.enable()

handler = StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(handler)

# ---------------- Set up data / config paths ----------------#
example_adt_data_dir = "/home/louy/Calibration_data_link/Atek/2024_05_07_EfmDataTest/adt_data_example/Apartment_release_clean_seq134/1WM103600M1292_optitrack_release_clean_seq134"
example_ase_data_dir = "/home/louy/Calibration_data_link/Atek/2024_05_07_EfmDataTest/ase_data_example/euston_simulation_100077_device0"
adt_config_path = os.path.join(
    "/home/louy/Calibration_data_link/Atek/2024_08_05_DryRun/adt_cubercnn_preprocess_config.yaml"
)
ase_config_path = os.path.join(
    "/home/louy/Calibration_data_link/Atek/2024_08_05_DryRun/ase_cubercnn_preprocess_config.yaml"
)
adt_to_atek_category_mapping_file = (
    "/home/louy/atek_on_fbsource/data/adt_prototype_to_atek.csv"
)
ase_to_atek_category_mapping_file = "/home/louy/atek_on_fbsource/data/ase_to_atek.csv"
output_wds_path = (
    "/home/louy/Calibration_data_link/Atek/2024_08_05_DryRun/wds_output/ase_test"
)
sequence_name = example_ase_data_dir.split("/")[-1]

# ---------------- Perform ATEK data preprocessing ----------------#
conf = OmegaConf.load(ase_config_path)
atek_preprocessor = create_general_atek_preprocessor_from_conf(
    conf=conf,
    raw_data_folder=example_ase_data_dir,
    sequence_name=sequence_name,
    output_wds_folder=output_wds_path,
    output_viz_file=None,
    category_mapping_file=ase_to_atek_category_mapping_file,
)

atek_preprocessor.process_all_samples(write_to_wds_flag=True, viz_flag=True)

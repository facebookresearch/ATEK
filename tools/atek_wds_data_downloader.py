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


import argparse
import logging

from atek.data_download.atek_data_store_download import download_atek_wds_sequences

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s:%(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(),  # Output logs to console
    ],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_args():
    parser = argparse.ArgumentParser(
        description="Parse dataverse urls to generate yaml file for training, validation and combined"
    )
    parser.add_argument("--config-name", type=str, help="Configuration name")
    parser.add_argument(
        "--input-json-path", type=str, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--train-val-split-ratio",
        type=float,
        default=0.9,
        help="Train-validation split ratio",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for shuffling data"
    )
    parser.add_argument(
        "--output-folder-path",
        type=str,
        help="Output folder path for YAML files and downloaded wds",
    )
    parser.add_argument(
        "--download-wds-to-local", action="store_true", help="Download WDS files"
    )
    parser.add_argument(
        "--max-num-sequences", type=int, help="Maximum number of sequences"
    )
    parser.add_argument(
        "--train-val-split-json-path",
        type=str,
        help="Path to the train validation split JSON file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    download_atek_wds_sequences(
        args.config_name,
        args.input_json_path,
        args.train_val_split_ratio,
        args.random_seed,
        args.output_folder_path,
        args.max_num_sequences,
        args.download_wds_to_local,
        args.train_val_split_json_path,
    )

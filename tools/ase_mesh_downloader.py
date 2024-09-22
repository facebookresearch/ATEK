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
import hashlib
import json
import logging
import os
import shutil
import zipfile

import requests
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s:%(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(),  # Output logs to console
    ],
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def download_ase_mesh(input_json_path: str, output_folder_path: str) -> None:
    with open(input_json_path, "r") as f:
        urls = json.load(f)

    # NOTE: by default it puts the .ply meshes under `ase_mesh` (need ~14G).
    # change this path if you want to put the mesh under a different folder
    os.makedirs(output_folder_path, exist_ok=True)

    logger.info(f"{len(urls)} plys to download")

    for url in tqdm.tqdm(urls):
        filename = url["filename"]
        cdn = url["cdn"]
        sha = url["sha"]

        # Download the file from the CDN
        response = requests.get(cdn)
        with open(filename, "wb") as f:
            f.write(response.content)

        # Check if the shasum matches
        with open(filename, "rb") as f:
            file_sha = hashlib.sha1(f.read()).hexdigest()
        if file_sha != sha:
            print(f"Error: Shasum mismatch for {filename}, {file_sha} != {sha}")
        else:
            print(f"Downloaded {filename} successfully")

        # Unzip the file
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall()

        filename_ply = os.path.splitext(filename)[0] + ".ply"

        # Move the unzipped file to the output folder
        shutil.move(filename_ply, output_folder_path)
        os.remove(filename)

    logger.info(f"Downloading done")


def get_args():
    parser = argparse.ArgumentParser(
        description="Download mesh files for ASE evaluation"
    )
    parser.add_argument(
        "--input-json-path", type=str, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output-folder-path",
        type=str,
        help="Output folder path for YAML files and downloaded wds",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    download_ase_mesh(
        args.input_json_path,
        args.output_folder_path,
    )

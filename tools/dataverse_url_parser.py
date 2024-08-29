import argparse
import json
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_tar_urls(json_data, config_name) -> Dict[str, List[str]]:
    wds_file_urls = json_data["atek_data_for_all_configs"][config_name]["wds_file_urls"]
    result = {}
    for sequence, shards in wds_file_urls.items():
        result[sequence] = []
        for shard, details in shards.items():
            if "download_url" in details:
                result[sequence].append(details["download_url"])
    return result


def main(
    config_name: str,
    input_json_path: str,
    output_folder_path: str,
):
    assert os.path.exists(
        input_json_path
    ), f"Input JSON file {input_json_path} does not exist."
    assert (
        os.path.exists(output_folder_path) is False
    ), f"Output folder {output_folder_path} already exists."
    with open(input_json_path, "r") as f:
        json_data = json.load(f)
    tar_urls = extract_tar_urls(json_data, config_name)


def get_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--config-name", type=str, help="Configuration name")
    parser.add_argument(
        "--input-json-path", type=str, help="Path to the input JSON file"
    )
    parser.add_argument(
        "--output-folder-path", type=str, help="Output folder path for YAML files"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.config_name,
        args.input_json_path,
        args.output_folder_path,
    )

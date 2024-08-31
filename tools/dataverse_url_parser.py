import argparse
import json
import logging
import os
import random
from typing import Dict, List, Optional

import requests
import yaml
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s:%(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(),  # Output logs to console
    ],
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_yaml(data: Dict, file_path: str):
    import yaml

    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def split_data(tar_urls: dict, ratio: float, seed: int):
    """
    Given a dictionary of tar URLs, split it into train and validation sets based on the specified ratio and random seed.
    """
    random.seed(seed)
    sequences = list(tar_urls.keys())
    sequences.sort()
    random.shuffle(sequences)
    split_point = int(len(sequences) * ratio)
    train_sequences = sequences[:split_point]
    valid_sequences = sequences[split_point:]
    train_data = {sequence: tar_urls[sequence] for sequence in train_sequences}
    valid_data = {sequence: tar_urls[sequence] for sequence in valid_sequences}
    return train_data, valid_data


def extract_tar_urls(
    wds_file_urls: Dict, max_num_sequences: Optional[int]
) -> Dict[str, List[str]]:
    result = {}
    cur_sequence_num = 0
    for sequence, shards in wds_file_urls.items():
        if (
            max_num_sequences
            and max_num_sequences >= 0
            and cur_sequence_num >= max_num_sequences
        ):
            break
        result[sequence] = []
        for shard, details in shards.items():
            if "download_url" in details:
                result[sequence].append(details["download_url"])
        cur_sequence_num += 1
    return result


def download_files(urls: List[str], output_dir: str) -> List[str]:
    """
    For each sequence, download all the files to a directory with the same name.
    Return a dictionary mapping sequence names to lists of downloaded file paths.
    """
    tar_paths = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Set up retry strategy
    retries = Retry(
        total=5,
        backoff_factor=2,  # final try is 2*(2**5) = 64 seconds
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    http = requests.Session()
    http.mount("http://", adapter)
    http.mount("https://", adapter)
    failed_urls = []
    for url in urls:
        # example url: https://scontent.xx.fbcdn.net/m1/v/t6/xxx.tar/AriaSyntheticEnvironment_1_0_ATEK_cubercnn_ase_simulation_0_device0_shards-0004.tar?xxx
        # Extract the filename from the URL
        # example filename: AriaSyntheticEnvironment_1_0_ATEK_cubercnn_ase_simulation_0_device0_shards-0004.tar
        filename = url.split("/")[-1].split("?")[0].split("_")[-1]
        filepath = os.path.join(output_dir, filename)
        # Initialize the list for this sequence if not already done
        try:
            response = http.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for bad status codes
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            tar_paths.append(os.path.join(output_dir.split("/")[-1], filename))
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            failed_urls.append(url)
    if failed_urls:
        logger.error(f"Failed to download {len(failed_urls)}")
    return tar_paths


def main(
    config_name: str,
    input_json_path: str,
    train_val_split_ratio: float,
    random_seed: int,
    output_folder_path: str,
    max_num_sequences: Optional[int] = None,
    download_wds_to_local: bool = False,
):
    assert os.path.exists(
        input_json_path
    ), f"Input JSON file {input_json_path} does not exist."
    assert (
        os.path.exists(output_folder_path) is False
    ), f"Output folder {output_folder_path} already exists."

    with open(input_json_path, "r") as file:
        data = json.load(file)
    wds_file_urls = data["atek_data_for_all_configs"][config_name]["wds_file_urls"]
    tar_urls = extract_tar_urls(wds_file_urls, max_num_sequences)

    train_tars, valid_tars = split_data(tar_urls, train_val_split_ratio, random_seed)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    if download_wds_to_local is True:
        # Download files
        new_train_tars = {}
        new_valid_tars = {}
        for sequence_name, urls in train_tars.items():
            logger.info(f"Downloading {sequence_name}")
            sequence_dir = os.path.join(output_folder_path, sequence_name)
            os.makedirs(sequence_dir)
            tar_paths = download_files(urls, sequence_dir)
            new_train_tars[sequence_name] = tar_paths

        for sequence_name, urls in valid_tars.items():
            logger.info(f"Downloading {sequence_name}")
            sequence_dir = os.path.join(output_folder_path, sequence_name)
            os.makedirs(sequence_dir)
            tar_paths = download_files(urls, sequence_dir)
            new_valid_tars[sequence_name] = tar_paths
        new_all_tars = new_train_tars | new_valid_tars
        # Save the YAML files
        with open(
            os.path.join(output_folder_path, "local_train_tars.yaml"), "w"
        ) as file:
            yaml.dump({"tars": new_train_tars}, file, default_flow_style=False)
        with open(
            os.path.join(output_folder_path, "local_validation_tars.yaml"), "w"
        ) as file:
            yaml.dump({"tars": new_valid_tars}, file, default_flow_style=False)
        with open(os.path.join(output_folder_path, "local_all_tars.yaml"), "w") as file:
            yaml.dump({"tars": new_all_tars}, file, default_flow_style=False)
        logger.info("Downloaded all files successfully.")

    else:
        with open(
            os.path.join(output_folder_path, "streamable_train_tars.yaml"), "w"
        ) as file:
            yaml.dump({"tars": train_tars}, file, default_flow_style=False)
        with open(
            os.path.join(output_folder_path, "streamable_validation_tars.yaml"), "w"
        ) as file:
            yaml.dump({"tars": valid_tars}, file, default_flow_style=False)
        with open(
            os.path.join(output_folder_path, "streamable_all_tars.yaml"), "w"
        ) as file:
            yaml.dump({"tars": tar_urls}, file, default_flow_style=False)


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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(
        args.config_name,
        args.input_json_path,
        args.train_val_split_ratio,
        args.random_seed,
        args.output_folder_path,
        args.max_num_sequences,
        args.download_wds_to_local,
    )

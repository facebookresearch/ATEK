from argparse import Namespace
from typing import Dict

from torch.utils.data import Dataset

from atek.dataset.cubercnn_data import AtekCubercnnInferDataset
from atek.dataset.omni3d_adapter import create_omni3d_webdataset


def create_dataset_config(args: Namespace, model_config) -> Dict:
    model_name = model_config["model_name"]

    if model_name == "cubercnn":
        dataset_config = {
            "data_type": args.data_type,
            "model_name": args.model_name,
            "category_id_remapping_json": model_config["cubercnn_cfg"].ID_MAP_JSON,
        }
    else:
        raise ValueError(f"Unknown model name for building dataset: {model_name}")

    return dataset_config


def create_inference_dataset(data_path: str, dataset_config: Dict) -> Dataset:
    """
    Create dataset for inference pipeline.

    Args:
        data_path (str): path to a sequence
        dataset_config (Dict): options to create dataset, such as model_name

    Return:
        dataset (Dataset): a dataset instance for the inference pipeline
    """
    model_name = dataset_config["model_name"]
    data_type = dataset_config["data_type"]

    if model_name == "cubercnn":
        if data_type == "wds":
            dataset = create_omni3d_webdataset(
                [data_path],
                batch_size=1,
                repeat=False,
                category_id_remapping_json=dataset_config["category_id_remapping_json"],
            )
        elif data_type == "raw":
            dataset = AtekCubercnnInferDataset(data_path, selected_device_number=0)
        else:
            raise ValueError(
                f"Unknown input data type for building dataset: {data_type}"
            )
    else:
        raise ValueError(f"Unknown model name for building dataset: {model_name}")

    return dataset

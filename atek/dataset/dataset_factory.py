from typing import Dict

from torch.utils.data import Dataset

from atek.dataset.cubercnn_data import AtekCubercnnInferDataset
from atek.dataset.omni3d_adapter import create_omni3d_webdataset


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
    if model_name == "cubercnn":
        dataset = AtekCubercnnInferDataset(data_path, selected_device_number=0)
    else:
        raise ValueError(f"Unknown model name for building dataset: {model_name}")
    return dataset

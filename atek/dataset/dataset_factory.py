from argparse import Namespace
from torch.utils.data import Dataset

from atek.dataset.cubercnn_data import AtekCubercnnInferDataset


def create_inference_dataset(data_path: str, args: Namespace) -> Dataset:
    """
    Create dataset for inference pipeline.

    Args:
        data_path (str): path to a sequence
        args (Namespace): arguments with model architecture name and other options

    Return:
        dataset (Dataset): a dataset instance for the inference pipeline
    """
    if args.model_name == "cubercnn":
        dataset = AtekCubercnnInferDataset(data_path, selected_device_number=0)
    else:
        raise ValueError(f"Unknown model name for building dataset: {args.model_name}")
    return dataset

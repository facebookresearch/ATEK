from argparse import Namespace
from typing import Dict

from torch.utils.data import Dataset

from atek.dataset.omni3d_adapter import (
    create_omni3d_raw_dataset,
    create_omni3d_webdataset,
)


def create_inference_dataset(
    data_path: str, args: Namespace, model_config: Dict
) -> Dataset:
    """
    Create dataset for inference pipeline.

    Args:
        data_path (str): path to a sequence
        args (Namespace): args with options related to dataset creation
        model_config (Dict): model config with options related to dataset creation

    Return:
        dataset (Dataset): a dataset instance for the inference pipeline
    """

    if args.model_name == "cubercnn":
        if args.data_type == "wds":
            dataset = create_omni3d_webdataset(
                [data_path],
                batch_size=1,
                repeat=False,
                category_id_remapping_json=model_config["cubercnn_cfg"].ID_MAP_JSON,
            )
        elif args.data_type == "raw":
            dataset = create_omni3d_raw_dataset(
                data_path, selected_device_number=0, target_color_format="BGR"
            )
        else:
            raise ValueError(
                f"Unknown input data type for building dataset: {args.data_type}"
            )
    else:
        raise ValueError(f"Unknown model name for building dataset: {args.model_name}")

    return dataset

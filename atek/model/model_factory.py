from argparse import Namespace
from typing import Callable, Dict
from torch.utils.data import Dataset
from atek.model.cubercnn import (
    create_cubercnn_model,
    convert_cubercnn_prediction,
    save_cubercnn_prediction,
)
from atek.viz.cubercnn_viz import AtekCubercnnInferViewer


def create_inference_model(args: Namespace):
    """
    Create the model for inference pipeline.

    Args:
        args (Namespace): arguments with model architecture name and other options

    Returns:
        cfg: model-specific config
        model: model used for the inference pipeline
    """
    if args.model_name == "cubercnn":
        cfg, model = create_cubercnn_model(args)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_name}")

    return cfg, model


def create_inference_callback(dataset: Dataset, callback_config: Dict) -> Dict[str, Callable]:
    """
    Create callback functions for the inference pipeline, such as pre-/post-processing
    of model input/output, visualizing and saving model predictions. Callback functions
    are setup based on model arhitecture and dataset requirements.

    Args:
        dataset (Dataset): dataset used for the inference pipeline
        callback_config (Dict): config for creating callbacks
    Returns:
        callbacks (Dict[str, Callable]): a dict of callback functions
    """
    model_name = callback_config["model_name"]

    if model_name == "cubercnn":
        iter_callbacks = []
        if callback_config["viewer"]["visualize"]:
            viewer = AtekCubercnnInferViewer(dataset, callback_config["viewer"])
            iter_callbacks.append(viewer)
        callbacks = {
            "iter_postprocess": convert_cubercnn_prediction,
            "iter_callback": iter_callbacks,
            "seq_callback": [save_cubercnn_prediction],
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {model_name}"
        )
    return callbacks

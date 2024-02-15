from argparse import Namespace
from typing import Callable, Dict

from atek.model.cubercnn import (
    CubercnnPredictionConverter,
    CubercnnPredictionSaver,
    create_cubercnn_config,
    create_cubercnn_model,
)
from atek.viz.cubercnn_viz import AtekCubercnnInferViewer


def create_model_config(args: Namespace) -> Dict:
    if args.model_name == "cubercnn":
        model_config = create_cubercnn_config(args)
    else:
        raise ValueError(
            f"Unknown model architecture for creating model_config: {args.model_name}"
        )
    return model_config


def create_inference_model(model_config: Dict):
    """
    Create the model for inference pipeline.

    Args:
        model_config (Dict): dict-based detailed model configurations

    Returns:
        model: model used for the inference pipeline
    """
    model_name = model_config["model_name"]
    if model_name == "cubercnn":
        model = create_cubercnn_model(model_config)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    return model


def create_callback_config(args: Namespace, model_config: Dict):
    if args.model_name == "cubercnn":
        callback_config = {
            "data_type": args.data_type,
            "model_name": args.model_name,
            "viewer": {
                "visualize": args.visualize,
                "web_port": args.web_port,
                "ws_port": args.ws_port,
                "camera_name": "camera_rgb",
            },
            "postprocessor": {
                "score_threshold": args.threshold,
                "category_names": model_config["cubercnn_cfg"].DATASETS.CATEGORY_NAMES,
            },
            "saver": {
                "output_dir": args.output_dir,
                "metadata_file": args.metadata_file,
                "prototype_file": args.prototype_file,
                "bbox3d_csv": args.bbox3d_csv,
                "category_id_remapping_json": model_config["cubercnn_cfg"].ID_MAP_JSON,
            },
        }
    else:
        raise ValueError(
            f"Unknown model architecture for creating callback_config: {args.model_name}"
        )

    return callback_config


def create_inference_callback(callback_config: Dict) -> Dict[str, Callable]:
    """
    Create callback functions for the inference pipeline, such as pre-/post-processing
    of model input/output, visualizing and saving model predictions. Callback functions
    are setup based on model arhitecture and dataset requirements.

    Args:
        callback_config (Dict): config for creating callbacks

    Returns:
        callbacks (Dict[str, Callable]): a dict of callback functions
    """
    model_name = callback_config["model_name"]
    data_type = callback_config["data_type"]

    if model_name == "cubercnn":
        iter_callbacks = []
        if callback_config["viewer"]["visualize"]:
            iter_callbacks.append(AtekCubercnnInferViewer(callback_config["viewer"]))

        seq_callbacks = []
        if data_type == "raw":
            seq_callbacks.append(CubercnnPredictionSaver(callback_config["saver"]))

        callbacks = {
            "iter_postprocess": CubercnnPredictionConverter(
                callback_config["postprocessor"]
            ),
            "iter_callback": iter_callbacks,
            "seq_callback": seq_callbacks,
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {model_name}"
        )

    return callbacks

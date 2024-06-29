from argparse import Namespace

from atek_v1.model.cubercnn import create_cubercnn_config, create_cubercnn_model


def create_inference_model(args: Namespace):
    """
    Create the model for inference pipeline, with the model config.

    Args:
        args (Namespace):

    Returns:
        model_config: dict-based detailed model configurations
        model: model used for the inference pipeline
    """
    if args.model_name == "cubercnn":
        model_config = create_cubercnn_config(args)
        model = create_cubercnn_model(model_config)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_name}")

    return model_config, model

from argparse import Namespace

from atek.model.cubercnn import create_cubercnn_inference_model


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
        model_config, model = create_cubercnn_inference_model(args)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_name}")

    return model_config, model

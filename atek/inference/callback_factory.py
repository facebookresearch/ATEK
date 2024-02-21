from argparse import Namespace
from typing import Callable, Dict, List, Union

from atek.inference.adt_prediction_saver import AdtPredictionSaver
from atek.inference.cubercnn_postprocessor import CubercnnPredictionConverter
from atek.viz.visualization_callbacks import AtekInferViewer


def create_inference_callback(
    args: Namespace, model_config: Dict
) -> Dict[str, Union[Callable, List[Callable]]]:
    """
    Create callback functions for the inference pipeline, such as visualization and saving model
    predictions, based on model arhitecture and dataset requirements.

    Args:
        args (Namespace): args with options to create callbacks, such as args.visualize
        model_config (Dict): model configs with options to creating the callbacks

    Returns:
        callbacks (Dict[str, Union[Callable, List[Callable]]): a dict of callback functions.
            `post_processor` converts model output to another format. `iteration_callbacks` are
            called after each iteration, while `sequence_callbacks` are called at the end of each
            webdataset tar or raw video sequence.
    """

    if args.model_name == "cubercnn":
        post_processor = CubercnnPredictionConverter(
            model_config["score_threshold"], model_config["category_names"]
        )

        iteration_callbacks = []
        if args.visualize:
            iteration_callbacks.append(
                AtekInferViewer(args.web_port, args.ws_port, "camera_rgb")
            )

        sequence_callbacks = []
        if args.data_type == "raw" or args.data_type == "wds":
            sequence_callbacks.append(
                AdtPredictionSaver(
                    args.output_dir,
                    args.metadata_file,
                    model_config["cfg"].ID_MAP_JSON,
                )
            )
        else:
            raise ValueError(
                f"Unknown input data type for creating sequence callbacks: {args.data_type}"
            )

        callbacks = {
            "post_processor": post_processor,
            "iteration_callbacks": iteration_callbacks,
            "sequence_callbacks": sequence_callbacks,
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {args.model_name}"
        )

    return callbacks

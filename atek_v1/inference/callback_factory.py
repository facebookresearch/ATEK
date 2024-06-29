from argparse import Namespace
from typing import Callable, Dict, List, Union

from atek_v1.evaluation.bbox3d_evaluator import Bbox3DEvaluator
from atek_v1.inference.adt_prediction_saver import AdtPredictionSaver
from atek_v1.inference.cubercnn_postprocessor import CubercnnPredictionConverter
from atek_v1.viz.visualization_callbacks import AtekInferViewer


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
            `post_processor` converts model output to another format. `per_batch_callbacks` are
            called after each iteration, while `per_sequence_callbacks` are called at the end of each
            webdataset tar or raw video sequence.
    """

    if args.model_name == "cubercnn":
        post_processor = CubercnnPredictionConverter(
            model_config["score_threshold"], model_config["category_names"]
        )

        per_batch_callbacks = []
        if args.visualize:
            per_batch_callbacks.append(
                AtekInferViewer(args.web_port, args.ws_port, "camera_rgb")
            )

        per_sequence_callbacks = []
        if args.save_prediction:
            per_sequence_callbacks.append(
                AdtPredictionSaver(
                    args.output_dir,
                    args.metadata_file,
                    model_config["cfg"].ID_MAP_JSON,
                )
            )

        eval_callbacks = []
        if args.evaluate:
            eval_callback = Bbox3DEvaluator(args.metric_name, args.metric_threshold)
            eval_callbacks.append(eval_callback)

        callbacks = {
            "post_processor": post_processor,
            "per_batch_callbacks": per_batch_callbacks,
            "per_sequence_callbacks": per_sequence_callbacks,
            "eval_callbacks": eval_callbacks,
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {args.model_name}"
        )

    return callbacks

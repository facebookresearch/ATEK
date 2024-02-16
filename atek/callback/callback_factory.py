from argparse import Namespace
from typing import Callable, Dict

from atek.evaluation.dump_cubercnn_results import CubercnnPredictionSaver
from atek.viz.cubercnn_viz import AtekCubercnnInferViewer


def create_inference_callback(
    args: Namespace, model_config: Dict
) -> Dict[str, Callable]:
    """
    Create callback functions for the inference pipeline, such as visualization and
    saving model predictions, based on model arhitecture and dataset requirements.

    Args:
        args (Namespace): args with options to create callbacks, such as args.visualize
        model_config (Dict): model configs with options to c

    Returns:
        callbacks (Dict[str, Callable]): a dict of callback functions
    """

    if args.model_name == "cubercnn":
        iter_callbacks = []
        if args.visualize:
            viewer_config = {
                "web_port": args.web_port,
                "ws_port": args.ws_port,
                "camera_name": "camera_rgb",
            }
            iter_callbacks.append(AtekCubercnnInferViewer(viewer_config))

        seq_callbacks = []
        if args.data_type == "raw":
            saver_config = {
                "output_dir": args.output_dir,
                "metadata_file": args.metadata_file,
                "category_id_remapping_json": model_config["cubercnn_cfg"].ID_MAP_JSON,
            }
            seq_callbacks.append(CubercnnPredictionSaver(saver_config))

        callbacks = {
            "iter_callback": iter_callbacks,
            "seq_callback": seq_callbacks,
        }
    else:
        raise ValueError(
            f"Unknown model architecture for setting callbacks: {args.model_name}"
        )

    return callbacks

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import fields

import torch

import tqdm
import yaml

from atek.data_loaders.atek_wds_dataloader import load_atek_wds_dataset

from atek.data_loaders.cubercnn_model_adaptor import (
    cubercnn_collation_fn,
    CubeRCNNModelAdaptor,
    load_atek_wds_dataset_as_cubercnn,
)
from atek.data_preprocess.atek_data_sample import (
    create_atek_data_sample_from_flatten_dict,
)

from atek.viz.atek_visualizer import NativeAtekSampleVisualizer

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import build_model  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import launch

from omegaconf import DictConfig, OmegaConf


def create_inference_model(config_file, ckpt_dir):
    """
    Create the model for inference pipeline, with the model config.
    """
    # Create default model configuration
    model_config = get_cfg()
    get_cfg_defaults(model_config)

    # add extra configs for data
    model_config.MAX_TRAINING_ATTEMPTS = 3
    model_config.TRAIN_LIST = ""
    model_config.TEST_LIST = ""
    model_config.ID_MAP_JSON = ""
    model_config.OBJ_PROP_JSON = ""
    model_config.CATEGORY_JSON = ""
    model_config.DATASETS.OBJECT_DETECTION_MODE = ""
    model_config.SOLVER.VAL_MAX_ITER = 0

    model_config.merge_from_file(config_file)
    model_config.freeze()

    model = build_model(model_config, priors=None)

    _ = DetectionCheckpointer(model, save_dir=ckpt_dir).resume_or_load(
        model_config.MODEL.WEIGHTS, resume=True
    )
    model.eval()

    return model_config, model


def get_tars(tar_yaml, use_relative_path=False):
    with open(tar_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
    if use_relative_path:
        data_dir = os.path.dirname(tar_yaml)
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    return tar_files


def run_inference(args):
    # parse in config file
    conf = OmegaConf.load(args.config_file)

    # setup config and model
    model_config, model = create_inference_model(args.config_file, args.ckpt_dir)

    # set up data loader from eval dataset
    infer_tars = get_tars(args.input_wds_tar, use_relative_path=True)
    # For testing only
    infer_tars = infer_tars[:10]

    infer_wds_atek_format = load_atek_wds_dataset(
        urls=infer_tars,
        batch_size=1,
        repeat_flag=False,
        collation_fn=cubercnn_collation_fn,
    )
    infer_wds_cubercnn_format = load_atek_wds_dataset_as_cubercnn(
        urls=infer_tars, batch_size=1, repeat_flag=False
    )

    atek_format_dataloader = torch.utils.data.DataLoader(
        infer_wds_atek_format,
        batch_size=None,
        num_workers=conf.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )
    cubercnn_format_dataloader = torch.utils.data.DataLoader(
        infer_wds_cubercnn_format,
        batch_size=None,
        num_workers=conf.DATALOADER.NUM_WORKERS,
        pin_memory=True,
    )

    # Loop over all batched samples in the dataset
    atek_viz = NativeAtekSampleVisualizer()
    # Setup profiling
    num_iters = 0
    num_samples = 0
    start_time = time.time()

    for atek_native_data, data in tqdm.tqdm(
        zip(atek_format_dataloader, cubercnn_format_dataloader),
        desc="Inference progress: ",
    ):
        num_iters += 1
        model_output = model(data)

        # Batched samples are formulated as lists in CubeRCNN, hence fill predictions over the lists
        for single_atek_input, single_cubercnn_input, single_cubercnn_output in zip(
            atek_native_data, data, model_output
        ):
            num_samples += 1
            gt_sample = create_atek_data_sample_from_flatten_dict(single_atek_input)
            prediction_in_atek_format = CubeRCNNModelAdaptor.cubercnn_gt_to_atek_gt(
                cubercnn_pred_dict=single_cubercnn_output,
                T_world_camera_np=single_cubercnn_input["T_world_camera"],
                camera_label="camera-rgb",
            )

            # Filter out prediction with low confidence scores
            # TODO: refactor this. Very ugly
            valid_viz_filter = prediction_in_atek_format["scores"] > 0.8
            for key, val in prediction_in_atek_format["obb2_gt"]["camera-rgb"].items():
                if isinstance(val, torch.Tensor):
                    prediction_in_atek_format["obb2_gt"]["camera-rgb"][key] = val[
                        valid_viz_filter
                    ]

            for key, val in prediction_in_atek_format["obb3_gt"]["camera-rgb"].items():
                if isinstance(val, torch.Tensor):
                    prediction_in_atek_format["obb3_gt"]["camera-rgb"][key] = val[
                        valid_viz_filter
                    ]

            # Visualize the GT results
            atek_viz.plot_atek_sample(
                atek_data_sample=gt_sample,
                plot_line_color=NativeAtekSampleVisualizer.COLOR_GREEN,
                suffix="_gt",
            )

            # Visualize the prediction results, only visualize the GT part
            atek_viz.plot_gtdata(
                atek_gt_dict=prediction_in_atek_format,
                timestamp_ns=gt_sample.camera_rgb.capture_timestamps_ns.item(),
                plot_line_color=NativeAtekSampleVisualizer.COLOR_RED,
                suffix="_infer",
            )

    # print time information
    elapsed_time = time.time() - start_time
    profile_message = (
        f"Inference time: {elapsed_time:.3f} secs for {num_iters} iters, "
        + f"{elapsed_time / num_iters:.2f} sec/iter, "
        + f"{num_samples} total samples "
        + f"{elapsed_time / num_samples:.2f} sec/sample"
    )
    print(profile_message)


def get_args():
    """
    Parse input arguments from CLI.
    """
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-wds-tar", default=None, help="Input WDS tar files to run inference on."
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory to save model predictions"
    )
    parser.add_argument(
        "--ckpt-dir", default=None, help="Directory for model checkpoint"
    )
    parser.add_argument(
        "--config-file",
        default=None,
        metavar="FILE",
        help="Path to Omega config file",
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    print("Command Line Args:", args)

    launch(
        run_inference,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        args=(args,),
    )

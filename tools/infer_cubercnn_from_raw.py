# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import logging
import os
import sys
import time

import torch

from atek.data_loaders.atek_raw_dataloader_as_cubercnn import (
    AtekRawDataloaderAsCubercnn,
)

from atek.data_loaders.cubercnn_model_adaptor import CubeRCNNModelAdaptor

from atek.viz.atek_visualizer import NativeAtekSampleVisualizer

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import build_model  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg

from detectron2.engine import default_setup, launch

from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,  # Set the minimum log level to DEBUG
    format="%(asctime)s-%(levelname)s:%(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(),  # Output logs to console
    ],
)

logger = logging.getLogger("infer_cubercnn")


def create_inference_model(config_file, ckpt_dir, use_cpu_only=False):
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
    if use_cpu_only:
        model_config.MODEL.DEVICE = "cpu"
    model_config.freeze()

    model = build_model(model_config, priors=None)

    _ = DetectionCheckpointer(model, save_dir=ckpt_dir).resume_or_load(
        model_config.MODEL.WEIGHTS, resume=True
    )
    model.eval()

    return model_config, model


def run_inference(args):
    # parse in config file
    conf = OmegaConf.load(args.config_file)

    # setup config and model
    model_config, model = create_inference_model(
        args.config_file, args.ckpt_dir, args.num_gpus == 0
    )

    # set up raw data loader from eval dataset
    preprocess_conf = OmegaConf.load(args.preprocess_config_file)

    raw_data_loader = AtekRawDataloaderAsCubercnn(
        conf=preprocess_conf,
        vrs_file=os.path.join(args.input_sequence_path, "video.vrs"),
        mps_files={
            "mps_closedloop_traj_file": os.path.join(
                args.input_sequence_path, "aria_trajectory.csv"
            ),
        },
    )

    # Loop over all samples in the dataset
    atek_viz = NativeAtekSampleVisualizer()
    # Setup profiling
    num_iters = 0
    for i in range(len(raw_data_loader)):
        num_iters += 1
        timestamps = raw_data_loader.get_timestamps_by_sample_index(i)

        # CubeRCNN subsampler should only return a single timestamp
        assert len(timestamps) == 1
        time_ns = timestamps[0]
        cubercnn_input_sample = (
            raw_data_loader.get_model_specific_sample_at_timestamp_ns(time_ns)
        )
        # skip if no sample is retrieved
        if cubercnn_input_sample is None:
            continue

        model_output = model([cubercnn_input_sample])

        atek_input_sample = raw_data_loader.get_atek_sample_at_timestamp_ns(time_ns)
        prediction_in_atek_format = CubeRCNNModelAdaptor.cubercnn_gt_to_atek_gt(
            cubercnn_pred_dict=model_output[0],
            T_world_camera_np=cubercnn_input_sample["T_world_camera"],
            camera_label="camera-rgb",
        )

        # Visualization
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
            atek_data_sample=atek_input_sample,
            plot_line_color=NativeAtekSampleVisualizer.COLOR_GREEN,
            suffix="_gt",
        )

        # Visualize the prediction results, only visualize the GT part
        atek_viz.plot_gtdata(
            atek_gt_dict=prediction_in_atek_format,
            timestamp_ns=time_ns,
            plot_line_color=NativeAtekSampleVisualizer.COLOR_RED,
            suffix="_infer",
        )


def get_args():
    """
    Parse input arguments from CLI.
    """
    parser = argparse.ArgumentParser(
        epilog=None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-sequence-path",
        default=None,
        help="Input sequence folder to run inference on.",
    )
    parser.add_argument(
        "--preprocess-config-file",
        default=None,
        help="ATEK preprocess configuration yaml file.",
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

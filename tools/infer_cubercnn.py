import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from dataclasses import fields
from typing import Dict

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
from atek.evaluation.static_object_detection.obb3_csv_io import GroupAtekObb3CsvWriter
from atek.util.atek_constants import ATEK_CATEGORY_NAME_TO_ID
from atek.util.file_io_utils import load_category_mapping_from_csv

from atek.viz.atek_visualizer import NativeAtekSampleVisualizer

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import build_model  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import launch

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


def get_tars(tar_yaml, use_relative_path=False):
    with open(tar_yaml, "r") as f:
        tar_files = yaml.safe_load(f)["tars"]
    if use_relative_path:
        data_dir = os.path.dirname(tar_yaml)
        tar_files = [os.path.join(data_dir, x) for x in tar_files]
    return tar_files


def parse_cubercnn_id_to_atek_id_mapping(
    model_config_file: str, ase_to_atek_file: str
) -> Dict[int, int]:
    """
    Helper function to handle the case where model output category mapping is different from ATEK's taxonomy.
    Model output taxonomy is stored under the model config.yaml file
    """
    with open(model_config_file, "r") as f:
        data = yaml.safe_load(f)
    # Extract the category names
    cat_mapping = data["DATASETS"]["CATEGORY_NAMES"]
    # Create a dictionary with IDs starting from 0
    cubercnn_id_to_name = {i: name for i, name in enumerate(cat_mapping)}

    cubercnn_id_to_atek_id = {}
    # Load in ase_to_atek name mapping
    ase_to_atek_mapping = load_category_mapping_from_csv(
        category_mapping_csv_file=ase_to_atek_file
    )

    for i, name in cubercnn_id_to_name.items():
        atek_name, atek_id = ase_to_atek_mapping[name]
        if atek_name not in ATEK_CATEGORY_NAME_TO_ID:
            raise ValueError(
                f"category name {name} mapped to {atek_name}, still not in ATEK mapping!"
            )

        cubercnn_id_to_atek_id[int(i)] = int(atek_id)

    return cubercnn_id_to_atek_id


def run_inference(args):
    # parse in config file
    conf = OmegaConf.load(args.config_file)

    # setup config and model
    model_config, model = create_inference_model(
        args.config_file, args.ckpt_dir, args.num_gpus == 0
    )

    # (Optional) parse in category mapping from model output to ATEK
    # TODO: this is hard-coded, remove this
    """
    cubercnn_id_to_atek_id = parse_cubercnn_id_to_atek_id_mapping(
        args.config_file,
        ase_to_atek_file="/home/louy/Calibration_data_link/Atek/2024_07_30_EvalTest/ase_to_atek.csv",
    )
    """

    # set up data loaders from eval dataset
    infer_tars = get_tars(args.input_wds_tar, use_relative_path=True)
    infer_wds_atek_format = load_atek_wds_dataset(
        urls=infer_tars,
        batch_size=args.batch_size,
        repeat_flag=False,
        collation_fn=cubercnn_collation_fn,
    )
    infer_wds_cubercnn_format = load_atek_wds_dataset_as_cubercnn(
        urls=infer_tars, batch_size=args.batch_size, repeat_flag=False
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

    # set up visualization
    if args.viz_flag:
        atek_viz = NativeAtekSampleVisualizer()

    # set up prediction writer
    pred_writer = GroupAtekObb3CsvWriter(
        output_folder=args.output_dir, output_filename="prediction_obbs.csv"
    )
    gt_writer = GroupAtekObb3CsvWriter(
        output_folder=args.output_dir, output_filename="gt_obbs.csv"
    )

    # Setup profiling
    num_iters = 0
    num_samples = 0
    start_time = time.time()
    current_sequence_name = ""

    # Loop over all batched samples in the dataset
    for atek_native_data, data in tqdm.tqdm(
        zip(atek_format_dataloader, cubercnn_format_dataloader),
        desc="Inference progress: ",
    ):
        num_iters += 1

        # Log time
        if num_iters % 100 == 0:
            end_time = time.time()
            logger.info(f"Time for iteration {num_iters} is {end_time - start_time}s")
            end_time = start_time

        try:
            model_output = model(data)

            # Batched samples are formulated as lists in CubeRCNN, hence fill predictions over the lists
            for single_atek_input, single_cubercnn_input, single_cubercnn_output in zip(
                atek_native_data, data, model_output
            ):
                num_samples += 1
                gt_sample = create_atek_data_sample_from_flatten_dict(single_atek_input)
                timestamp_ns = gt_sample.camera_rgb.capture_timestamps_ns.item()
                prediction_in_atek_format = CubeRCNNModelAdaptor.cubercnn_gt_to_atek_gt(
                    cubercnn_pred_dict=single_cubercnn_output,
                    T_world_camera_np=single_cubercnn_input["T_world_camera"],
                    camera_label="camera-rgb",
                    # cubercnn_id_to_atek_id=cubercnn_id_to_atek_id,
                )

                # Some printing info
                if current_sequence_name != single_atek_input["sequence_name"]:
                    current_sequence_name = single_atek_input["sequence_name"]
                    logger.info(
                        f"Starting to process sequence: {current_sequence_name}"
                    )

                # --------------------- Update CsvWriter ------------------------#
                gt_writer.write_from_atek_dict(
                    atek_dict=single_atek_input["gt_data"]["obb3_gt"]["camera-rgb"],
                    timestamp_ns=timestamp_ns,
                    sequence_name=single_atek_input["sequence_name"],
                )

                pred_writer.write_from_atek_dict(
                    atek_dict=prediction_in_atek_format["obb3_gt"]["camera-rgb"],
                    confidence_score=prediction_in_atek_format["scores"],
                    timestamp_ns=timestamp_ns,
                    sequence_name=single_atek_input["sequence_name"],
                )

                # --------------------- Update Viz ------------------------#
                # Filter out prediction with low confidence scores
                if args.viz_flag:
                    # TODO: refactor this. Very ugly
                    valid_viz_filter = prediction_in_atek_format["scores"] > 0.8
                    for key, val in prediction_in_atek_format["obb2_gt"][
                        "camera-rgb"
                    ].items():
                        if isinstance(val, torch.Tensor):
                            prediction_in_atek_format["obb2_gt"]["camera-rgb"][key] = (
                                val[valid_viz_filter]
                            )

                    for key, val in prediction_in_atek_format["obb3_gt"][
                        "camera-rgb"
                    ].items():
                        if isinstance(val, torch.Tensor):
                            prediction_in_atek_format["obb3_gt"]["camera-rgb"][key] = (
                                val[valid_viz_filter]
                            )

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

        except Exception as e:
            logger.error(
                f"processing failed for iter {num_iters}, for either {current_sequence_name} or the sequence after this. Skipping"
                f"error message: {e}"
            )
            # put pred_samples' category name and confidence together in obb2_gt and obb3_gt
            for i in range(
                len(
                    prediction_in_atek_format["obb2_gt"]["camera-rgb"]["category_names"]
                )
            ):
                prediction_in_atek_format["obb2_gt"]["camera-rgb"]["category_names"][
                    i
                ] += f": {prediction_in_atek_format['scores'][i]:.2f}"
            for i in range(
                len(
                    prediction_in_atek_format["obb3_gt"]["camera-rgb"]["category_names"]
                )
            ):
                prediction_in_atek_format["obb3_gt"]["camera-rgb"]["category_names"][
                    i
                ] += f": {prediction_in_atek_format['scores'][i]:.2f}"
    gt_writer.flush()
    pred_writer.flush()

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
        "--viz-flag",
        action="store_true",
        help=" Flag to control if show visualization (default: False)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=" Batch size to run inference in GPU",
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

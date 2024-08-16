# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import argparse
import json
import logging
import os

import numpy as np
import torch

from atek.evaluation.static_object_detection.eval_obb3 import (
    evaluate_obb3_for_single_csv_pair,
    evaluate_obb3_over_a_dataset,
)
from atek.evaluation.static_object_detection.eval_obb3_metrics_utils import (
    print_obb3_metrics_to_logger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s:%(message)s",  # Format of the log messages
    handlers=[
        logging.StreamHandler(),  # Output logs to console
    ],
)

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ATEK static 3D object detection benchmarking"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help=f"The input folder that contains the gt and pred obbs csv files."
        f" If this is provided, the eval will be done at dataset-level",
        default=None,
    )
    parser.add_argument(
        "--pred-csv",
        type=str,
        help="The prediction obbs csv file",
        default=None,
    )
    parser.add_argument(
        "--gt-csv",
        type=str,
        help="The ground truth obbs csv file",
        default=None,
    )
    parser.add_argument("--output-file", type=str, help="The output metrics file. ")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-num-sequences",
        type=int,
        default=-1,
    )
    args = parser.parse_args()

    if args.input_folder is not None:
        logger.info(f"Running dataset-level eval on {args.input_folder}")
        metrics = evaluate_obb3_over_a_dataset(
            input_folder=args.input_folder,
            gt_filename=args.gt_csv,
            prediction_filename=args.pred_csv,
            iou=args.iou_threshold,
            compute_per_class_metrics=False,
            max_num_sequences=args.max_num_sequences,
        )
        logger.info(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        assert (
            args.pred_csv is not None and args.gt_csv is not None
        ), "Either --input-folder or (--pred-csv+--gt-csv) must be provided"
        logger.info(f"Running file-level eval on {args.pred_csv} and {args.gt_csv}")
        metrics = evaluate_obb3_for_single_csv_pair(
            pred_csv=args.pred_csv,
            gt_csv=args.gt_csv,
            iou=args.iou_threshold,
            log_last_frame_result=False,
            compute_per_class_metrics=True,
        )
        print_obb3_metrics_to_logger(metrics)

    # Write metrics results to file
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict


import argparse
import json
import logging
import os

import numpy as np
import torch
from atek.evaluation.static_object_detection.eval_obb3_metrics_utils import (
    print_obb3_metrics_to_logger,
)

from atek.evaluation.surface_reconstruction.surface_reconstruction_metrics import (
    evaluate_mesh_over_a_dataset,
    evaluate_single_mesh_pair,
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
        description="Run ATEK surface reconstruction benchmarking"
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        help=f"The input folder that contains the gt and pred mesh `.ply` files."
        f" If this is provided, the eval will be done at dataset-level",
        default=None,
    )
    parser.add_argument(
        "--pred-mesh",
        type=str,
        help="The prediction mesh ply file",
        default=None,
    )
    parser.add_argument(
        "--gt-mesh",
        type=str,
        help="The ground truth mesh ply file",
        default=None,
    )
    parser.add_argument("--output-file", type=str, help="The output metrics file. ")
    parser.add_argument(
        "--is-adt-mesh",
        action="store_true",
        help=" Flag to indicate if the input mesh is from ADT dataset. If so, gravity correction will be applied (default: False)",
    )

    args = parser.parse_args()

    if args.input_folder is not None:
        logger.info(f"Running dataset-level eval on {args.input_folder}")
        metrics = evaluate_mesh_over_a_dataset(
            input_folder=args.input_folder,
            pred_mesh_filename=args.pred_mesh,
            gt_mesh_filename=args.gt_mesh,
        )
        logger.info(json.dumps(metrics, indent=2, sort_keys=True))
    else:
        assert (
            args.pred_mesh is not None and args.gt_mesh is not None
        ), "Either --input-folder or (--pred-mesh+--gt-mesh) must be provided"

        logger.info(f"Running file-level eval on {args.pred_mesh} and {args.gt_mesh}")
        metrics, _, _ = evaluate_single_mesh_pair(
            pred_mesh_filename=args.pred_mesh,
            gt_mesh_filename=args.gt_mesh,
            correct_mesh_gravity=args.is_adt_mesh,
        )

        # for testing only
        logger.info(f" metrics results is {metrics}")
        print_obb3_metrics_to_logger(metrics)

    # Write metrics results to file
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

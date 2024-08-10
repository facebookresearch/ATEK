# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import logging
from typing import Dict, Optional

import numpy as np

import torch
from atek.evaluation.static_object_detection.eval_obb3_metrics_utils import (
    prec_recall_bb3,
)

from atek.evaluation.static_object_detection.obb3_csv_io import AtekObb3CsvReader
from atek.evaluation.static_object_detection.static_object_detection_metrics import (
    AtekObb3Metrics,
)

from atek.util.atek_constants import ATEK_CATEGORY_ID_TO_NAME

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: make this a input from file
sem_id_to_name = ATEK_CATEGORY_ID_TO_NAME


def compute_prec_recall_for_single_timestamp(
    pred_obb_dict: Dict,
    gt_obb_dict: Dict,
    iou: float = 0.2,
    compute_per_class_metrics: bool = False,
) -> Dict:
    """
    Compute precision and recall for a single timestamp, given the prediction and gt obbs.
    """
    result = {}

    # perform precision recall calculation for single timestamp
    precision, recall, match_matrix, iou_matrix, per_class_results = prec_recall_bb3(
        pred_obb_dict,
        gt_obb_dict,
        iou_thres=iou,
        return_ious=True,
        per_class=compute_per_class_metrics,
    )

    # Compute stats including true positives, false positives from precision and recall
    true_positives = match_matrix.any(-1)

    # Log results into a dict
    result[f"precision@IoU{iou}"] = float(precision)
    result[f"recall@IoU{iou}"] = float(recall)
    result[f"num_true_positives@IoU{iou}"] = int(true_positives.sum())
    result["num_predictions"] = match_matrix.shape[0]
    result["num_groundtruth"] = match_matrix.shape[1]

    # Log per-class results
    if compute_per_class_metrics:
        for sem_id, per_class_result in per_class_results.items():
            result[f"precision@IoU{iou}@Class_{sem_id_to_name[sem_id.item()]}"] = float(
                per_class_result["precision"]
            )
            result[f"recall@IoU{iou}@Class_{sem_id_to_name[sem_id.item()]}"] = float(
                per_class_result["recall"]
            )

    return result


def update_from_single_sequence_obb3(
    pred_obb_dict: Dict,
    gt_obb_dict: Dict,
    mAP_3d: AtekObb3Metrics,
    iou: float,
    single_timestamp_to_log_result: Optional[int] = None,
    compute_per_class_metrics: bool = False,
) -> Dict:
    """
    Core function to evaluate a single pair of obb gt vs prediction, for a single sequence,
    and update the mAP_3d metrics accordingly.

    @param pred_obb_dict: a dict of {timestamp -> Obb3GtDict} for the prediction obbs, need to include "confidence_scores"
    @param gt_obb_dict: a dict of {timestamp -> Obb3GtDict} for the gt
    @param mAP_3d: the mAP_3d metrics to be updated.
    @param single_timestamp_to_log_result: if not None, log the precision and recall results for the single frame specified by this timestamp.
    """
    result = {}

    # Aggregate all timestamps
    all_timestamps = list(pred_obb_dict.keys()) + list(gt_obb_dict.keys())
    all_timestamps = list(set(all_timestamps))
    all_timestamps.sort()

    # Count for timestamps with empty GT and empty predictions
    empty_gt_timestamp_count = 0
    empty_pred_timestamp_count = 0
    for time in all_timestamps:
        if time not in pred_obb_dict:
            logger.info(f"prediction obbs not found for timestamp {time}")
            empty_pred_timestamp_count += 1
            continue
        if time not in gt_obb_dict:
            logger.info(f"gt obbs not found for {time}")
            empty_gt_timestamp_count += 1
            continue

        # check if the preds contain confidence scores
        confidence_score = pred_obb_dict[time]["confidence_scores"].squeeze()
        assert not torch.all(
            confidence_score.eq(-1.0)
        ), "the obbs don't contain valid confidence scores for mAP calculation."

        # TODO: check paddings are removed for EFM data

        # log precision-recall statistics for a single frame if needed
        if (
            single_timestamp_to_log_result is not None
            and time == single_timestamp_to_log_result
        ):
            single_timestamp_prec_recall_result = (
                compute_prec_recall_for_single_timestamp(
                    pred_obb_dict[time],
                    gt_obb_dict[time],
                    iou=iou,
                    compute_per_class_metrics=compute_per_class_metrics,
                )
            )
            result.update(single_timestamp_prec_recall_result)

        # add pred/gt pair to mAP calculator.
        mAP_3d.update(pred_obb_dict[time], gt_obb_dict[time])

    # End for loop over timestamps
    result["num_timestamps"] = len(all_timestamps)
    result["num_timestamp_miss_pred"] = empty_pred_timestamp_count
    result["num_timestamp_miss_gt"] = empty_gt_timestamp_count

    return result


def evaluate_obb3_for_single_csv_pair(
    pred_csv: str,
    gt_csv: str,
    iou: float = 0.2,
    log_last_frame_result: bool = False,
    compute_per_class_metrics: bool = False,
) -> Dict:
    """
    Evaluate a single pair of prediction and gt obbs csv files, and return a dict of metrics including mean average precision (mAP), precision, recall based on IOU value.
    """
    # Read in prediction and gt obbs
    pred_reader = AtekObb3CsvReader(input_filename=pred_csv)
    pred_obb_dict = pred_reader.read_as_obb_dict()
    gt_reader = AtekObb3CsvReader(input_filename=gt_csv)
    gt_obb_dict = gt_reader.read_as_obb_dict()

    # TODO: check or allow different taxonomies from prediction and gt

    result = {}
    mAP_3d = AtekObb3Metrics(
        class_metrics=compute_per_class_metrics,
        global_name_to_id={
            name: int(sem_id) for sem_id, name in sem_id_to_name.items()
        },
    )

    # If last frame result needs to be logged
    timestamp_to_log = None
    if log_last_frame_result:
        timestamp_to_log = max(pred_obb_dict.keys()) if pred_obb_dict else None

    single_sequence_result = update_from_single_sequence_obb3(
        pred_obb_dict=pred_obb_dict,
        gt_obb_dict=gt_obb_dict,
        mAP_3d=mAP_3d,
        iou=iou,
        compute_per_class_metrics=compute_per_class_metrics,
    )
    result.update(single_sequence_result)

    # Compute mAP metrics numbers, ignore average recall (e.g. "mar_*")
    result_map = mAP_3d.compute()
    result_map = {
        k: v.item() for k, v in result_map.items() if not k.startswith("mar_")
    }
    result.update(result_map)

    return result

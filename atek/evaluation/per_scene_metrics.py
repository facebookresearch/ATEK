# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import Union

import numpy as np
import pandas as pd
import torch
from atek.evaluation import iou_giou
from atek.utils.obb3 import Obb3


METRIC_DATA_TYPES = {
    "CategoryId": int,
    "PredId": int,
    "GtId": int,
}


def flatten_to_list(torch_tensor):
    return torch_tensor.flatten().numpy().tolist()


def compute_per_scene_metrics(
    obb3_pred: Union[Obb3, None], obb3_gt: Union[Obb3, None]
) -> pd.DataFrame:
    """
    Compute per-scene pairwise metrics between predicted and GT Obb3's. For example, if there are
    2 predictions and 4 GTs for of the same category, this function will return a dataframe with
    2x4=8 rows. Each row corresponds to one prediction-GT pair. The metrics include intersection
    volume, IoU, and GIoU between each pair of GT and predicted Obb3's.

    Args:
        obb3_pred: Predicted Obb3's
        obb3_gt: Ground truth Obb3's

    Returns:
        metrics_df: DataFrame containing per-scene pairwise metrics
    """

    if obb3_gt is None:
        pred_num = len(obb3_pred.instance_id)
        metrics = [
            {
                "CategoryId": flatten_to_list(obb3_pred.category_id),
                "PredId": flatten_to_list(obb3_pred.instance_id),
                "GtId": [-1] * pred_num,
                "Confidence": flatten_to_list(obb3_pred.score),
                "IoU": [np.nan] * pred_num,
                "GIoU": [np.nan] * pred_num,
            }
        ]
    elif obb3_pred is None:
        gt_num = len(obb3_gt.instance_id)
        metrics = [
            {
                "CategoryId": flatten_to_list(obb3_gt.category_id),
                "PredId": [-1] * gt_num,
                "GtId": flatten_to_list(obb3_gt.instance_id),
                "Confidence": [np.nan] * gt_num,
                "IoU": [np.nan] * gt_num,
                "GIoU": [np.nan] * gt_num,
            }
        ]
    else:
        category_ids = torch.cat((obb3_pred.category_id, obb3_gt.category_id)).unique()
        metrics = []

        for cat_id in category_ids:
            pred_idx = torch.nonzero(obb3_pred.category_id == cat_id).flatten()
            gt_idx = torch.nonzero(obb3_gt.category_id == cat_id).flatten()
            scores = obb3_pred.score[pred_idx]

            if cat_id not in obb3_pred.category_id:
                # False negative: set all metrics to NaN
                gt_num = len(gt_idx)
                curr_metrics = {
                    "CategoryId": [cat_id] * gt_num,
                    "PredId": [-1] * gt_num,
                    "GtId": flatten_to_list(obb3_gt.instance_id[gt_idx]),
                    "Confidence": [np.nan] * gt_num,
                    "IoU": [np.nan] * gt_num,
                    "GIoU": [np.nan] * gt_num,
                }
            elif cat_id not in obb3_gt.category_id:
                # False positive: set all metrics to NaN
                pred_num = len(pred_idx)
                curr_metrics = {
                    "CategoryId": [cat_id] * pred_num,
                    "PredId": flatten_to_list(obb3_pred.instance_id[pred_idx]),
                    "GtId": [-1] * pred_num,
                    "Confidence": flatten_to_list(obb3_pred.score[pred_idx]),
                    "IoU": [np.nan] * pred_num,
                    "GIoU": [np.nan] * pred_num,
                }
            else:
                # compute metrics
                _, iou, giou = iou_giou(
                    obb3_pred.size[pred_idx],
                    obb3_pred.T_ref_obj[pred_idx],
                    obb3_gt.size[gt_idx],
                    obb3_gt.T_ref_obj[gt_idx],
                )

                pred_id_grid, gt_id_grid = torch.meshgrid(
                    obb3_pred.instance_id[pred_idx], obb3_gt.instance_id[gt_idx]
                )
                confidence_grid, _ = torch.meshgrid(
                    scores, torch.Tensor([1] * len(gt_idx))
                )

                curr_metrics = {
                    "CategoryId": [cat_id] * len(pred_id_grid.flatten()),
                    "PredId": flatten_to_list(pred_id_grid),
                    "GtId": flatten_to_list(gt_id_grid),
                    "Confidence": flatten_to_list(confidence_grid),
                    "IoU": flatten_to_list(iou),
                    "GIoU": flatten_to_list(giou),
                }
            metrics.append(curr_metrics)

    metrics_df = pd.concat(
        [pd.DataFrame(item).astype(METRIC_DATA_TYPES) for item in metrics],
        ignore_index=True,
    )

    return metrics_df

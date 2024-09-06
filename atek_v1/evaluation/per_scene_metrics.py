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

from typing import Optional

import numpy as np
import pandas as pd
import torch
from atek.utils.obb3 import Obb3
from atek_v1.evaluation.bbox3d_metrics import diagonal_error, iou_giou
from atek_v1.evaluation.math_metrics.distance_metrics import (
    chamfer_distance,
    euclidean_distance,
    hungarian_distance,
)
from atek_v1.evaluation.math_metrics.rotation_metrics import geodesic_angular_error


METRIC_DATA_TYPES = {
    "category_id": int,
    "pred_id": int,
    "gt_id": int,
}


def flatten_to_list(torch_tensor):
    return torch_tensor.flatten().numpy().tolist()


def compute_per_scene_metrics(
    obb3_pred: Optional[Obb3] = None, obb3_gt: Optional[Obb3] = None
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

    if obb3_gt is None and obb3_pred is None:
        return pd.DataFrame()
    elif obb3_gt is None:
        pred_num = len(obb3_pred.instance_id)
        metrics = [
            {
                "category_id": flatten_to_list(obb3_pred.category_id),
                "pred_id": flatten_to_list(obb3_pred.instance_id),
                "gt_id": [-1] * pred_num,
                "confidence": flatten_to_list(obb3_pred.score),
                "iou": [np.nan] * pred_num,
                "giou": [np.nan] * pred_num,
                "box_corner_chamfer_distance": [np.nan] * pred_num,
                "box_corner_hungarian_distance": [np.nan] * pred_num,
                "center_euclidean_distance": [np.nan] * pred_num,
                "rotation_geodesic_error": [np.nan] * pred_num,
                "box_corner_diagonal_error": [np.nan] * pred_num,
            }
        ]
    elif obb3_pred is None:
        gt_num = len(obb3_gt.instance_id)
        metrics = [
            {
                "category_id": flatten_to_list(obb3_gt.category_id),
                "pred_id": [-1] * gt_num,
                "gt_id": flatten_to_list(obb3_gt.instance_id),
                "confidence": [np.nan] * gt_num,
                "iou": [np.nan] * gt_num,
                "giou": [np.nan] * gt_num,
                "box_corner_chamfer_distance": [np.nan] * gt_num,
                "box_corner_hungarian_distance": [np.nan] * gt_num,
                "center_euclidean_distance": [np.nan] * gt_num,
                "rotation_geodesic_error": [np.nan] * gt_num,
                "box_corner_diagonal_error": [np.nan] * gt_num,
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
                    "category_id": [cat_id] * gt_num,
                    "pred_id": [-1] * gt_num,
                    "gt_id": flatten_to_list(obb3_gt.instance_id[gt_idx]),
                    "confidence": [np.nan] * gt_num,
                    "iou": [np.nan] * gt_num,
                    "giou": [np.nan] * gt_num,
                    "box_corner_chamfer_distance": [np.nan] * gt_num,
                    "box_corner_hungarian_distance": [np.nan] * gt_num,
                    "center_euclidean_distance": [np.nan] * gt_num,
                    "rotation_geodesic_error": [np.nan] * gt_num,
                    "box_corner_diagonal_error": [np.nan] * gt_num,
                }
            elif cat_id not in obb3_gt.category_id:
                # False positive: set all metrics to NaN
                pred_num = len(pred_idx)
                curr_metrics = {
                    "category_id": [cat_id] * pred_num,
                    "pred_id": flatten_to_list(obb3_pred.instance_id[pred_idx]),
                    "gt_id": [-1] * pred_num,
                    "confidence": flatten_to_list(obb3_pred.score[pred_idx]),
                    "iou": [np.nan] * pred_num,
                    "giou": [np.nan] * pred_num,
                    "box_corner_chamfer_distance": [np.nan] * pred_num,
                    "box_corner_hungarian_distance": [np.nan] * pred_num,
                    "center_euclidean_distance": [np.nan] * pred_num,
                    "rotation_geodesic_error": [np.nan] * pred_num,
                    "box_corner_diagonal_error": [np.nan] * pred_num,
                }
            else:
                # compute metrics
                _, iou, giou = iou_giou(
                    obb3_pred.size[pred_idx],
                    obb3_pred.T_ref_obj[pred_idx],
                    obb3_gt.size[gt_idx],
                    obb3_gt.T_ref_obj[gt_idx],
                )
                chamfer_dist = chamfer_distance(
                    obb3_pred.bb3_in_ref_frame[pred_idx],
                    obb3_gt.bb3_in_ref_frame[gt_idx],
                )
                hungarian_dist = hungarian_distance(
                    obb3_pred.bb3_in_ref_frame[pred_idx],
                    obb3_gt.bb3_in_ref_frame[gt_idx],
                )
                euclidean_dist = euclidean_distance(
                    obb3_pred.t_ref_obj[pred_idx],
                    obb3_gt.t_ref_obj[gt_idx],
                )
                geodesic_ang_err = geodesic_angular_error(
                    obb3_pred.R_ref_obj[pred_idx],
                    obb3_gt.R_ref_obj[gt_idx],
                )
                diag_err = diagonal_error(
                    obb3_pred.size[pred_idx],
                    obb3_gt.size[gt_idx],
                )

                pred_id_grid, gt_id_grid = torch.meshgrid(
                    obb3_pred.instance_id[pred_idx], obb3_gt.instance_id[gt_idx]
                )
                confidence_grid, _ = torch.meshgrid(
                    scores, torch.Tensor([1] * len(gt_idx))
                )

                curr_metrics = {
                    "category_id": [cat_id] * len(pred_id_grid.flatten()),
                    "pred_id": flatten_to_list(pred_id_grid),
                    "gt_id": flatten_to_list(gt_id_grid),
                    "confidence": flatten_to_list(confidence_grid),
                    "iou": flatten_to_list(iou),
                    "giou": flatten_to_list(giou),
                    "box_corner_chamfer_distance": flatten_to_list(chamfer_dist),
                    "box_corner_hungarian_distance": flatten_to_list(hungarian_dist),
                    "center_euclidean_distance": flatten_to_list(euclidean_dist),
                    "rotation_geodesic_error": flatten_to_list(geodesic_ang_err),
                    "box_corner_diagonal_error": flatten_to_list(diag_err),
                }
            metrics.append(curr_metrics)

    metrics_df = pd.concat(
        [pd.DataFrame(item).astype(METRIC_DATA_TYPES) for item in metrics],
        ignore_index=True,
    )

    return metrics_df

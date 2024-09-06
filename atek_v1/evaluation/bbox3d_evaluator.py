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

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from atek_v1.evaluation.per_scene_metrics import (
    compute_per_scene_metrics,
    METRIC_DATA_TYPES,
)
from atek_v1.utils.eval_utils import compute_average_precision
from atek_v1.utils.obb3 import Obb3


def _set_false_positive(fp_df: pd.DataFrame) -> pd.DataFrame:
    false_positive_df = fp_df.copy()
    false_positive_df.loc[:, "gt_id"] = -1
    for col in false_positive_df.columns:
        if col not in [
            "category_id",
            "pred_id",
            "gt_id",
            "confidence",
        ]:
            false_positive_df.loc[:, col] = np.nan
    return false_positive_df


def _set_false_negative(fn_df: pd.DataFrame) -> pd.DataFrame:
    false_negative_df = fn_df.copy()
    false_negative_df.loc[:, "pred_id"] = -1
    for col in false_negative_df.columns:
        if col not in [
            "category_id",
            "pred_id",
            "gt_id",
        ]:
            false_negative_df.loc[:, col] = np.nan
    return false_negative_df


class Bbox3DEvaluator:
    """
    Evaluator for 3D bounding boxes
    """

    def __init__(
        self,
        metric_name: str = "IoU",
        metric_thresh: float = 0.25,
    ):
        """
        Args:
            metric_name (str): name of the metric used to match predicted and GT 3D boxes, such as IoU
                GIoU
            metric_thresh (float): threshold to decide if metric between predicted and GT is enough to
                be considered a good match.
        """
        self.metric_name = metric_name
        self.metric_thresh = metric_thresh
        if metric_name in ["IoU", "GIoU"]:
            self.metric_higher = True
        else:
            self.metric_higher = False
        self.matched_metrics_df_list = []

    def update(self, model_input: List[Dict], model_prediction: List[Dict]) -> None:
        """
        Compute 3D bounding box match metrics (IoU, GIoU, etc) for the model input and prediction
        pair, accumulate the results
        """
        for input, prediction in zip(model_input, model_prediction):
            if len(input["object_dimensions"]) == 0:
                obb3_gt = None
            else:
                dim_gt = input["object_dimensions"]
                t_world_obj_gt = input["Ts_world_object"][:, :, 3]
                R_world_obj_gt = input["Ts_world_object"][:, :, :3]
                category_id_gt = input["instances"].gt_classes
                fake_instance_id_gt = torch.Tensor(list(range(len(category_id_gt))))

                obb3_gt = Obb3(
                    dim_gt,
                    t_world_obj_gt,
                    R_world_obj_gt,
                    fake_instance_id_gt,
                    category_id_gt,
                )

            N_pred = len(prediction)
            if N_pred == 0:
                obb3_pred = None
            else:
                dim_pred = torch.Tensor([pred["dimensions"] for pred in prediction])
                t_world_obj_pred = torch.zeros((N_pred, 3))
                R_world_obj_pred = torch.zeros((N_pred, 3, 3))
                T_world_cam_4x4 = torch.eye(4)
                T_world_cam_4x4[:3, :] = input["T_world_camera"]
                for i, pred in enumerate(prediction):
                    T_cam_obj_pred_4x4 = torch.eye(4)
                    T_cam_obj_pred_4x4[:3, 3] = torch.Tensor(pred["t_cam_obj"])
                    T_cam_obj_pred_4x4[:3, :3] = torch.Tensor(pred["R_cam_obj"])
                    T_world_obj_pred_4x4 = T_world_cam_4x4 @ T_cam_obj_pred_4x4

                    t_world_obj_pred[i, :] = T_world_obj_pred_4x4[:3, 3]
                    R_world_obj_pred[i, :, :] = T_world_obj_pred_4x4[:3, :3]
                category_id_pred = torch.Tensor(
                    [pred["category_idx"] for pred in prediction]
                )
                fake_instance_id_pred = torch.Tensor(list(range(len(category_id_pred))))
                score_pred = torch.Tensor([pred["score"] for pred in prediction])
                obb3_pred = Obb3(
                    dim_pred,
                    t_world_obj_pred,
                    R_world_obj_pred,
                    fake_instance_id_pred,
                    category_id_pred,
                    score_pred,
                )

            # Compute per-scene (frame/frameset/frameset group) pairwise metrics (IoU/GIoU/Chamfer distance/etc)
            metrics_df = compute_per_scene_metrics(obb3_pred, obb3_gt)
            if len(metrics_df) == 0:
                continue
            # Match predicted boxes to GT boxes
            matched_metrics_df = self.match_pred_to_gt(metrics_df)

            matched_metrics_df.insert(0, "timestamp_ns", input["timestamp_ns"])

            # Append the matched metrics for the final evaluation aggregation
            self.matched_metrics_df_list.append(matched_metrics_df)

    def match_pred_to_gt(self, per_scene_metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Match prediction 3D bounding boxes to ground truth 3D bounding boxes based on per-scene
        pairwise metrics, given a match criterion. For example, if the match criterion is IoU of
        threshold 0.25, then a predicted box with IoU over 0.25 to a GT box is considered a match. All
        the metrics will be returned for the matched pred and GT pair, such as IoU and GIoU.

        Args:
            per_scene_metrics_df (pd.DataFrame): DataFrame containing pred-to-GT metrics for a scene.

        Returns:
            matched_metrics_df (pd.DataFrame): DataFrame containing all metrics for each matched pred
                and GT boxes, and unmatched pred/GT boxes (metrics will be np.nan).
        """
        # Find false positives and false negatives
        false_positives = per_scene_metrics_df.query("gt_id == -1")
        false_negatives = per_scene_metrics_df.query("pred_id == -1")

        # Define condition for matching
        sign = ">=" if self.metric_higher else "<="
        match_condition = f"{self.metric_name} {sign} {self.metric_thresh}"

        # Process remaining predictions and GTs
        filtered_df = per_scene_metrics_df.query("gt_id != -1 & pred_id != -1")
        true_positives = []
        additional_false_positives = []
        additional_false_negatives = []

        # Append true positives
        matched_df = filtered_df.query(f"{match_condition}")

        if not matched_df.empty:
            if self.metric_higher:
                row_idx = matched_df.groupby("gt_id")[self.metric_name].idxmax()
            else:
                row_idx = matched_df.groupby("gt_id")[self.metric_name].idxmin()
            true_positives.append(matched_df.loc[row_idx])

        # Append additional false positives: preds not meeting match criterion
        matched_pred_ids = matched_df["pred_id"].unique().tolist()  # noqa
        curr_false_positives = filtered_df.query(
            "pred_id not in @matched_pred_ids"
        ).reset_index(drop=True)
        additional_false_positives.append(
            _set_false_positive(curr_false_positives).drop_duplicates()
        )

        # Append additional false negatives: GTs that have no valid matched preds,
        # i.e., preds with not meeting match criterion
        matched_gt_ids = matched_df["gt_id"].unique().tolist()  # noqa
        curr_false_negatives = filtered_df.query(
            "gt_id not in @matched_gt_ids"
        ).reset_index(drop=True)
        additional_false_negatives.append(
            _set_false_negative(curr_false_negatives).drop_duplicates()
        )

        if len(true_positives) > 0:
            true_positives = pd.concat(true_positives)
        if len(additional_false_positives) > 0:
            additional_false_positives = pd.concat(additional_false_positives)
        if len(additional_false_negatives) > 0:
            additional_false_negatives = pd.concat(additional_false_negatives)

        matched_metrics_df = [
            item
            for item in [
                false_positives,
                false_negatives,
                true_positives,
                additional_false_positives,
                additional_false_negatives,
            ]
            if len(item) > 0
        ]
        matched_metrics_df = (
            pd.concat(matched_metrics_df)
            .astype(METRIC_DATA_TYPES)
            .sort_values(by=["pred_id", "gt_id"])
            .reset_index(drop=True)
        )

        return matched_metrics_df

    def evaluate(self) -> Dict[str, Any]:
        """
        Use the accumulated results to run full evaluation
        """
        matched_metrics_df_all = pd.concat(self.matched_metrics_df_list)

        gt_count = matched_metrics_df_all.query("gt_id != -1").shape[0]
        if gt_count == 0:
            return {"mAP": np.nan}

        df = matched_metrics_df_all.query("pred_id != -1").sort_values(
            by=["confidence"], ascending=False
        )
        y_true = df["gt_id"].to_numpy() != -1
        if len(y_true) == 0:
            return {"mAP": 0}

        y_scores = df["confidence"].to_numpy()
        mean_ap = compute_average_precision(y_true, y_scores, gt_count)
        results = {"mAP": mean_ap}

        return results

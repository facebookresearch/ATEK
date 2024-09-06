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

from typing import Dict, List


class CubercnnPredictionConverter:
    """
    Convert CubeRCNN model predictions from detectron2 instance to a list of dicts

    Args:
        score_threshold (float): threshold to filter out low-confidence predictions
        category_names (List[str]): the list of predicted object categories.
    """

    def __init__(self, score_threshold: str, category_names: List[str]):
        self.score_threshold = score_threshold
        self.category_names = category_names

    def __call__(self, model_input: List[Dict], model_prediction: List[Dict]):
        """
        Converts per-frame CubeRCNN prediction to list of dicts
        """
        converted_model_predictions = []

        for input, prediction in zip(model_input, model_prediction):
            dets = prediction["instances"]
            preds_per_frame = []

            if len(dets) == 0:
                continue

            for (
                corners3D,
                center_cam,
                center_2D,
                dimensions,
                bbox_2D,
                pose,
                score,
                scores_full,
                cat_idx,
            ) in zip(
                dets.pred_bbox3D,
                dets.pred_center_cam,
                dets.pred_center_2D,
                dets.pred_dimensions,
                dets.pred_boxes,
                dets.pred_pose,
                dets.scores,
                dets.scores_full,
                dets.pred_classes,
            ):
                if score < self.score_threshold:
                    continue
                cat = self.category_names[cat_idx]
                predictions_dict = {
                    "sequence_name": input["sequence_name"],
                    "frame_id": input["frame_id"],
                    "timestamp_ns": input["timestamp_ns"],
                    "T_world_cam": input["T_world_camera"],
                    # CubeRCNN dimensions are in reversed order of Aria data convention
                    "dimensions": dimensions.tolist()[::-1],
                    "t_cam_obj": center_cam.tolist(),
                    "R_cam_obj": pose.tolist(),
                    "corners3D": corners3D.tolist(),
                    "center_2D": center_2D.tolist(),
                    "bbox_2D": bbox_2D.tolist(),
                    "score": score.detach().item(),
                    "scores_full": scores_full.tolist(),
                    "category_idx": cat_idx.detach().item(),
                    "category": cat,
                }
                preds_per_frame.append(predictions_dict)

            converted_model_predictions.append(preds_per_frame)

        return converted_model_predictions

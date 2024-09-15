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

import logging
import re
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from pytorch3d.ops.iou_box3d import (
    _box3d_overlap,
    _box_planes,
    _box_triangles,
    _check_nonzero,
)
from torch.nn import functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def coplanar_mask(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Determines if the vertices of the given boxes are coplanar.
    This function checks if the fourth vertex of each face of the boxes lies on the plane defined by the first three vertices.
    Parameters:
    - boxes (torch.Tensor): A tensor of shape (B, 8, 3) representing B boxes, each defined by 8 vertices in 3D space.
    - eps (float): A small threshold used to determine coplanarity. Defaults to 1e-4.
    Returns:
    - torch.Tensor: A boolean tensor of shape (B*P,) where each element indicates whether the corresponding face's vertices are coplanar.
    """
    faces = torch.tensor(_box_planes, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = F.normalize(v1 - v0, dim=-1)
    e1 = F.normalize(v2 - v0, dim=-1)
    normal = F.normalize(torch.cross(e0, e1, dim=-1), dim=-1)

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    good = (mat1.bmm(mat2).abs() < eps).view(-1)
    return good


def nonzero_area_mask(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    faces = torch.tensor(_box_triangles, dtype=torch.int64, device=boxes.device)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2
    return (face_areas > eps).all(-1)


def bb3_valid(boxes: torch.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the box is valid
    """
    # Check that the box is not degenerate
    return nonzero_area_mask(boxes, eps) & coplanar_mask(boxes, eps)


@dataclass
class IouOutputs:
    vol: torch.Tensor
    iou: torch.Tensor


def box3d_overlap_wrapper(
    boxes1: torch.Tensor, boxes2: torch.Tensor, eps: float = 1e-3
) -> IouOutputs:
    """
    only compute ious and volumes for good boxes and recompose with 0s for all bad boxes.
    its better because it can handle if a subset of boxes is bad. But it costs more compute.
    """
    if not all((8, 3) == box.shape[1:] for box in [boxes1, boxes2]):
        raise ValueError("Each box in the batch must be of shape (8, 3)")
    m1 = bb3_valid(boxes1, eps)
    m2 = bb3_valid(boxes2, eps)
    b1_good = boxes1[m1]
    b2_good = boxes2[m2]
    vol = torch.zeros(boxes1.shape[0], boxes2.shape[0], device=boxes1.device)
    iou = torch.zeros_like(vol)
    if b1_good.shape[0] == 0 or b2_good.shape[0] == 0:
        logger.debug("no valid bbs returning 0 volumes and ious")
    else:
        try:
            vol_good, iou_good = _box3d_overlap.apply(b1_good, b2_good)
            m_good = m1.unsqueeze(-1) & m2.unsqueeze(0)
            vol[m_good] = vol_good.view(-1)
            iou[m_good] = iou_good.view(-1)
        except Exception:
            logger.exception("returning 0 volumes and ious because of an exception")
    return IouOutputs(vol=vol, iou=iou)


def box3d_volume(boxes: torch.Tensor) -> torch.Tensor:
    """
    Computes the volume of a set of 3d bounding boxes.

    Args:
        boxes (Tensor[N, 8, 3]): 3d boxes for which the volume will be computed.

    Returns:
        Tensor[N]: the volume for each box
    """
    if boxes.numel() == 0:
        return torch.zeros(0).to(boxes)
    # Triple product to calculate volume
    a = boxes[:, 1, :] - boxes[:, 0, :]
    b = boxes[:, 3, :] - boxes[:, 0, :]
    c = boxes[:, 4, :] - boxes[:, 0, :]
    vol = torch.abs(torch.cross(a, b, dim=-1) @ c.T)[0]
    return vol


def remove_invalid_box3d(obbs: Dict) -> Dict:
    """
    Remove invalid bboxes
    """
    boxes = obbs["bbox_corners_in_world"]  # (num_obb, 8, 3)
    assert boxes.dim() == 3
    assert (8, 3) == boxes.shape[1:]
    valid_ind = []
    for b in range(boxes.shape[0]):
        try:
            # no need for co planarity check since our obbs are good by construction.
            # _check_coplanar(boxes[b : b + 1, :, :])
            _check_nonzero(boxes[b : b + 1, :, :])
            valid_ind.append(b)
        except Exception:
            pass

    # remove invalid obbs
    valid_obbs = {
        "instance_ids": obbs["instance_ids"][valid_ind].clone().detach(),
        "category_names": [
            obbs["category_names"][i] for i in valid_ind
        ],  # list needs to be explicitly constructed
        "category_ids": obbs["category_ids"][valid_ind].clone().detach(),
        "object_dimensions": obbs["object_dimensions"][valid_ind].clone().detach(),
        "ts_world_object": obbs["ts_world_object"][valid_ind].clone().detach(),
        "bbox_corners_in_world": obbs["bbox_corners_in_world"][valid_ind]
        .clone()
        .detach(),
        "confidence_scores": obbs["confidence_scores"][valid_ind].clone().detach(),
    }

    return valid_obbs


def prec_recall_bb3(
    padded_pred: Dict,
    padded_target: Dict,
    iou_thres=0.2,
    return_ious=False,
    per_class=False,
):
    """
    Computes precision and recall for 3D bounding boxes based on Intersection over Union (IoU).
    Parameters:
    - padded_pred (Dict): A dictionary containing predicted 3d bounding boxes, in the same format as ATEK's obb3_gt_processor.
    - padded_target (Dict): A dictionary containing ground truth 3d bounding boxes, in the same format as ATEK's obb3_gt_processor.
    - iou_thres (float, optional): The IoU threshold to consider a prediction as a true positive. Defaults to 0.2.
    - return_ious (bool, optional): If True, returns IoUs along with precision and recall. Defaults to False.
    - per_class (bool, optional): If True, computes precision and recall per class. Defaults to False.
    Returns:
    - tuple: A tuple containing precision, recall, a matching matrix, optionally IoUs, and optionally per class results.

    The function filters out invalid bounding boxes from the predictions and targets, computes the IoU for each pair of predicted and target boxes, and determines matches based on the IoU threshold and class labels. It calculates precision and recall based on these matches and can optionally return IoUs and per-class results.
    """
    # TODO: check if the obb tensors are padded

    # Remove invalid obbs
    original_pred_len = len(padded_pred["category_ids"])
    original_target_len = len(padded_target["category_ids"])

    pred = remove_invalid_box3d(padded_pred)
    target = remove_invalid_box3d(padded_target)
    new_pred_len = len(pred["category_ids"])
    new_target_len = len(target["category_ids"])
    if new_pred_len != original_pred_len:
        logging.warning(
            f"Warning: predicted obbs filtered from {original_pred_len} to {new_pred_len}"
        )
    if new_target_len != original_target_len:
        logging.warning(
            f"Warning: target obbs filtered from {original_target_len} to {new_target_len}"
        )

    prec_recall = (-1.0, -1.0, None)
    # deal with edge cases first
    if new_pred_len == 0:
        # invalid precision and 0 recall
        prec_recall = (-1.0, 0.0, None)
        return prec_recall
    elif new_target_len == 0:
        # invalid recall and 0 precision
        prec_recall = (0.0, -1.0, None)
        return prec_recall

    # Pred should be of shape [N, 1], target should be of shape [1, M]
    pred_sems = pred["category_ids"].unsqueeze(1)
    target_sems = target["category_ids"].unsqueeze(0)

    # 1. Match classes
    sem_id_match = pred_sems == target_sems
    # 2. Match IoUs
    ious = box3d_overlap_wrapper(
        pred["bbox_corners_in_world"], target["bbox_corners_in_world"]
    ).iou
    iou_match = ious > iou_thres

    # 3. Match both
    sem_iou_match = torch.logical_and(sem_id_match, iou_match)
    # make final matching matrix
    final_sem_iou_match = torch.zeros_like(sem_iou_match).bool()
    num_pred = sem_iou_match.shape[0]  # TP + FP
    num_target = sem_iou_match.shape[1]  # TP + FN
    # 4. Deal with the case where one prediction correspond to multiple GTs.
    # In this case, only the GT with highest IoU is considered the match.
    for pred_idx in range(int(num_pred)):
        if sem_iou_match[pred_idx, :].sum() <= 1:
            final_sem_iou_match[pred_idx, :] = sem_iou_match[pred_idx, :].clone()
        else:
            tgt_ious = ious[pred_idx, :].clone()
            tgt_ious[~sem_iou_match[pred_idx, :]] = -1.0
            sorted_ids = torch.argsort(tgt_ious, descending=True)
            tp_id = sorted_ids[0]
            # Set the pred with highest iou
            final_sem_iou_match[pred_idx, :] = False
            final_sem_iou_match[pred_idx, tp_id] = True

    # 5. Deal with the case where one GT correspond to multiple predictions.
    # In this case, if the predictions contain probabilities, we take the one with the highest score, otherwise, we take the one with the highest iou.
    for gt_idx in range(int(num_target)):
        if final_sem_iou_match[:, gt_idx].sum() <= 1:
            continue
        else:
            pred_scores = pred["confidence_scores"].squeeze(-1).clone()
            if torch.all(pred_scores.eq(-1.0)):
                # go with highest iou
                pred_ious = ious[:, gt_idx].clone()
                pred_ious[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_ious, descending=True)
                tp_id = sorted_ids[0]
                # Set the pred with highest iou
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True
            else:
                # go with the highest score
                pred_scores[~final_sem_iou_match[:, gt_idx]] = -1.0
                sorted_ids = torch.argsort(pred_scores, descending=True)
                tp_id = sorted_ids[0]
                final_sem_iou_match[:, gt_idx] = False
                final_sem_iou_match[tp_id, gt_idx] = True

    TPs = final_sem_iou_match.any(-1)
    # precision = TP / (TP + FP) = TP / #Preds
    num_tp = TPs.sum().item()
    prec = num_tp / num_pred
    # recall = TP / (TP + FN) = TP / #GTs
    rec = num_tp / num_target

    ret = (prec, rec, final_sem_iou_match)
    if return_ious:
        ret = ret + (ious,)
    else:
        ret = ret + (None,)

    if per_class:
        # per class prec and recalls
        per_class_results = {}
        all_sems = torch.cat([pred_sems.squeeze(-1), target_sems.squeeze(0)], dim=0)
        unique_classes = torch.unique(all_sems.squeeze(-1))
        for sem_id in unique_classes:
            pred_obbs_sem = pred_sems.squeeze(-1) == sem_id
            TPs_sem = (TPs & pred_obbs_sem).sum().item()
            num_pred_sem = pred_obbs_sem.sum().item()
            gt_obbs_sem = target_sems.squeeze(0) == sem_id
            num_gt_sem = gt_obbs_sem.sum().item()
            prec_sem = TPs_sem / num_pred_sem if num_pred_sem > 0 else -1.0
            rec_sem = TPs_sem / num_gt_sem if num_gt_sem > 0 else -1.0
            per_class_results[sem_id] = {}
            per_class_results[sem_id]["num_true_positives"] = TPs_sem
            per_class_results[sem_id]["num_dets"] = num_pred_sem
            per_class_results[sem_id]["num_gts"] = num_gt_sem
            per_class_results[sem_id]["precision"] = prec_sem
            per_class_results[sem_id]["recall"] = rec_sem
        ret = ret + (per_class_results,)
    else:
        ret = ret + (None,)

    return ret


def _extract_prec_recall_values_from_dict(metrics_results: Dict) -> Tuple:
    """
    A helper function to extract precision and recall values from a dictionary of metrics.
    """
    # Pattern to match keys starting with "Precision@IOU"
    # for testing only
    print(f"------- debug: {metrics_results.keys()}")
    prec_pattern = r"^precision@IoU"
    recall_pattern = r"^recall@IoU"

    prec = -1.0
    recall = -1.0
    for key, val in metrics_results.items():
        if re.match(prec_pattern, key):
            prec = val
        elif re.match(recall_pattern, key):
            recall = val
    return prec, recall


def print_obb3_metrics_to_logger(metrics) -> None:
    # Initialize an empty string to accumulate log messages
    log_messages = "Object Detection Model Performance Summary\n"
    log_messages += "=======Overall mAP Scores across all classes=======\n"
    log_messages += f"mAP (Average across IoU thresholds, defined by MeanAveragePrecision3D class, default is [0.05, 0.10, 0.15, ..., 0.5]): {metrics['map_3D']:.4f}\n"
    average_prec, average_recall = _extract_prec_recall_values_from_dict(metrics)
    log_messages += f"Average precision (IoU=0.20): {average_prec:.4f}\n"
    log_messages += f"Average recall (IoU=0.20): {average_recall:.4f}\n"
    log_messages += (
        "===mAP across IoU thresholds [0.05, 0.10, 0.15, ..., 0.5]) per Class===\n"
    )

    # Extract and sort the per-class mAP entries
    class_map = {
        key: value for key, value in metrics.items() if "map_per_class@" in key
    }
    sorted_class_map = sorted(
        class_map.items(), key=lambda item: item[1], reverse=True
    )  # Sort by value in ascending order

    for key, value in sorted_class_map:
        # Skip entries of -1
        if value < 0.0:
            continue
        class_name = key.split("@")[1].replace("_3D", "").replace("_", " ").title()
        log_messages += f"{class_name}: {value:.4f}\n"

    log_messages += "=======Timestamp Information=======\n"
    log_messages += f"Number of timestamps: {metrics['num_timestamps']}\n"
    log_messages += f"Number of timestamps with missing ground truth: {metrics['num_timestamp_miss_gt']}\n"
    log_messages += f"Number of timestamps with missing predictions: {metrics['num_timestamp_miss_pred']}\n"
    # Log the entire summary in one log statement
    logger.info(log_messages)

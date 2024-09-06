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
from dataclasses import dataclass
from time import time
from typing import Dict, List, Optional, Tuple

import torch

from atek.evaluation.static_object_detection.eval_obb3_metrics_utils import (
    bb3_valid,
    box3d_overlap_wrapper,
    box3d_volume,
)

from torchmetrics.detection.mean_ap import (
    _fix_empty_tensors,
    BaseMetricResults,
    MARMetricResults,
    MeanAveragePrecision,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MAPMetricResults3D(BaseMetricResults):
    """Class to wrap the final mAP results."""

    __slots__ = (
        "map",
        "map_25",
        "map_50",
        "map_small",
        "map_medium",
        "map_large",
    )


class AtekObb3Metrics(torch.nn.Module):
    """
    Metrics class that works with ATEK obb3 gt dict format.
    This is a thin wrapper around MeanAveragePrecision3D metrics class, to be able to use it with ATEK obb3 gt dict format.
    It is a torch.nn.Module to be able to behave like a torchmetrics object.

    This class is easily extensible to support more input format, by adding / overloading the `update()` API function.
    """

    def __init__(
        self,
        class_metrics: bool = False,
        global_name_to_id: Optional[Dict] = None,
        ret_all_prec_rec: Optional[bool] = False,
        max_detection_thresholds: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            cam_ids (list): list of camera ids to evaluate
            cam_names (list): list of camera names to evaluate
            class_metrics (bool): if True, computes per-class metrics
            max_detection_thresholds (list): list of maximum number of detections to evaluate
        """

        super().__init__()
        if max_detection_thresholds is None:
            # max number of detections to evaluate - 220 is sufficient for ASE scenes
            max_detection_thresholds = [220]

        self.metric_3d = MeanAveragePrecision3D(
            class_metrics=class_metrics,
            max_detection_thresholds=max_detection_thresholds,
            ret_all_prec_rec=ret_all_prec_rec,
        )
        self.sem_id_to_name = None

        if global_name_to_id is not None:
            self.sem_id_to_name = {
                int(sem_id): name for name, sem_id in global_name_to_id.items()
            }

    def update(self, prediction: Dict, target: Dict):
        """
        Update from prediction and target, in ATEK obb3 gt dict format.
        """
        # Compute bbox corners in world frame, as tensors of shape [N, 8, 3]
        pred_bb3corners_in_world = prediction["bbox_corners_in_world"]
        target_bb3corners_in_world = target["bbox_corners_in_world"]

        # convert to lists
        pred_list = [
            {
                "boxes": pred_bb3corners_in_world,
                "scores": prediction["confidence_scores"],
                "labels": prediction["category_ids"],
            }
        ]
        target_list = [
            {
                "boxes": target_bb3corners_in_world,
                "labels": target["category_ids"],
            }
        ]
        self.metric_3d.update(pred_list, target_list)

    def forward(self, prediction: Dict, target: Dict):
        self.update(prediction, target)
        return self.compute()

    def compute(self):
        # only compute the 3D metrics (mAP 3D)
        t0 = time()

        metrics = {}
        logger.info(f"Computing 3D obb metric")
        metrics_3d_result = self.metric_3d.compute(self.sem_id_to_name)

        t1 = time()
        logger.info(f"DONE Computing 3D obb metric in {t1-t0} seconds")

        for metric_name, val in metrics_3d_result.items():
            metrics[f"{metric_name}_3D"] = val
        return metrics

    def reset(self):
        for metric in self.metric_3d.values():
            metric.reset()


class MeanAveragePrecision3D(MeanAveragePrecision):
    def __init__(
        self,
        max_detection_thresholds: Optional[List[int]] = None,
        class_metrics: bool = False,  # compute per class metrics
        ret_all_prec_rec: bool = False,  # return all precision and recall values
    ) -> None:  # type: ignore
        # TODO: check for torchvision version

        # Set default IOU and recall thresholds
        iou_thresholds = torch.linspace(
            0.05, 0.5, round((0.5 - 0.05) / 0.05) + 1, dtype=torch.float64
        ).tolist()
        rec_thresholds = torch.linspace(
            0.0, 1.00, round(1.00 / 0.01) + 1, dtype=torch.float64
        ).tolist()
        super().__init__(
            iou_thresholds=iou_thresholds,
            rec_thresholds=rec_thresholds,
        )

        max_det_thr, _ = torch.sort(
            torch.IntTensor(max_detection_thresholds or [1, 10, 100])
        )
        self.max_detection_thresholds = max_det_thr.tolist()

        self.class_metrics = class_metrics

        # important to overwrite after the __init__() call since they are otherwise overwritten by super().__init__()
        self.bbox_area_ranges = {"all": (0, 1e5)}
        self.ret_all_prec_rec = ret_all_prec_rec
        self.eval_imgs = [] if self.ret_all_prec_rec else None

    def update(self, preds: List[Dict[str, torch.Tensor]], target: List[Dict[str, torch.Tensor]]) -> None:  # type: ignore
        """Add detections and ground truth to the metric.

        Args:
            preds: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 8, 3] containing `num_boxes` detection boxes of the format
                specified in the contructor. By default, this method expects
                (4) +---------+. (5)
                    | ` .     |  ` .
                    | (0) +---+-----+ (1)
                    |     |   |     |
                (7) +-----+---+. (6)|
                    ` .   |     ` . |
                    (3) ` +---------+ (2)
                box_corner_vertices = [
                    [xmin, ymin, zmin],
                    [xmax, ymin, zmin],
                    [xmax, ymax, zmin],
                    [xmin, ymax, zmin],
                    [xmin, ymin, zmax],
                    [xmax, ymin, zmax],
                    [xmax, ymax, zmax],
                    [xmin, ymax, zmax],
                ]
            - ``scores``: ``torch.FloatTensor`` of shape
                [num_boxes] containing detection scores for the boxes.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 0-indexed detection classes for the boxes.

            target: A list consisting of dictionaries each containing the key-values
            (each dictionary corresponds to a single image):
            - ``boxes``: ``torch.FloatTensor`` of shape
                [num_boxes, 8, 3] containing `num_boxes` ground truth boxes of the format
                specified in the contructor.
            - ``labels``: ``torch.IntTensor`` of shape
                [num_boxes] containing 1-indexed ground truth classes for the boxes.

        """
        # TODO: consider add input validation

        for item in preds:
            boxes = _fix_empty_tensors(item["boxes"])
            if hasattr(self, "detection_boxes"):
                self.detection_boxes.append(boxes)
            else:
                self.detections.append(boxes)

            self.detection_labels.append(item["labels"])
            self.detection_scores.append(item["scores"])

        for item in target:
            boxes = _fix_empty_tensors(item["boxes"])
            if hasattr(self, "groundtruth_boxes"):
                self.groundtruth_boxes.append(boxes)
            else:
                self.groundtruths.append(boxes)
            self.groundtruth_labels.append(item["labels"])

    def _compute_iou(self, id: int, class_id: int, max_det: int) -> torch.Tensor:
        """
        Overloading base class function.
        Computes the Intersection over Union (IoU) for ground truth and detection bounding boxes for the given
        image and class.

        Args:
            id:
                Image Id, equivalent to the index of supplied samples
            class_id:
                Class Id of the supplied ground truth and detection labels
            max_det:
                Maximum number of evaluated detection bounding boxes
        """
        if hasattr(self, "detection_boxes"):
            gt = self.groundtruth_boxes[id]
            det = self.detection_boxes[id]
        else:
            gt = self.groundtruths[id]
            det = self.detections[id]
        gt_label_mask = self.groundtruth_labels[id] == class_id
        det_label_mask = self.detection_labels[id] == class_id
        if len(gt_label_mask) == 0 or len(det_label_mask) == 0:
            return torch.Tensor([])
        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if len(gt) == 0 or len(det) == 0:
            return torch.Tensor([])

        # Sort by scores and use only max detections
        scores = self.detection_scores[id]
        scores_filtered = scores[self.detection_labels[id] == class_id]
        inds = torch.argsort(scores_filtered, descending=True)
        det = det[inds]
        if len(det) > max_det:
            det = det[:max_det]

        # generalized_box_iou
        # both det and gt are List of "boxes"
        ious = box3d_overlap_wrapper(det, gt).iou
        return ious

    def _evaluate_image(
        self,
        id: int,
        class_id: int,
        area_range: Tuple[int, int],
        max_det: int,
        ious: Dict,
    ) -> Optional[Dict]:
        """
        Overloading base class function.
        Perform evaluation for single class and image.

        Args:
            id:
                Image Id, equivalent to the index of supplied samples.
            class_id:
                Class Id of the supplied ground truth and detection labels.
            area_range:
                List of lower and upper bounding box area threshold.
            max_det:
                Maximum number of evaluated detection bounding boxes.
            ious:
                IoU results for image and class.
        """
        if hasattr(self, "detection_boxes"):
            gt = self.groundtruth_boxes[id]
            det = self.detection_boxes[id]
        else:
            gt = self.groundtruths[id]
            det = self.detections[id]

        gt_label_mask = self.groundtruth_labels[id] == class_id
        det_label_mask = self.detection_labels[id] == class_id
        if len(gt_label_mask) == 0 or len(det_label_mask) == 0:
            return None
        gt = gt[gt_label_mask]
        det = det[det_label_mask]
        if len(gt) == 0 and len(det) == 0:
            return None

        areas = box3d_volume(gt)
        ignore_area = (areas < area_range[0]) | (areas > area_range[1])

        # sort detection highest score first, sort gt ignore last
        ignore_area_sorted, gtind = torch.sort(ignore_area.to(torch.uint8))
        # Convert to uint8 temporarily and back to bool, because "Sort currently does not support bool dtype on CUDA"
        ignore_area_sorted = ignore_area_sorted.to(torch.bool)
        gt = gt[gtind]
        scores = self.detection_scores[id]
        scores_filtered = scores[det_label_mask]
        scores_sorted, dtind = torch.sort(scores_filtered, descending=True)
        det = det[dtind]
        if len(det) > max_det:
            det = det[:max_det]
        # load computed ious
        ious = (
            ious[id, class_id][:, gtind]
            if len(ious[id, class_id]) > 0
            else ious[id, class_id]
        )

        nb_iou_thrs = len(self.iou_thresholds)
        nb_gt = len(gt)
        nb_det = len(det)
        gt_matches = torch.zeros(
            (nb_iou_thrs, nb_gt), dtype=torch.bool, device=det.device
        )
        det_matches = torch.zeros(
            (nb_iou_thrs, nb_det), dtype=torch.bool, device=det.device
        )
        gt_ignore = ignore_area_sorted
        det_ignore = torch.zeros(
            (nb_iou_thrs, nb_det), dtype=torch.bool, device=det.device
        )

        if torch.numel(ious) > 0:
            for idx_iou, t in enumerate(self.iou_thresholds):
                for idx_det, _ in enumerate(det):
                    m = MeanAveragePrecision._find_best_gt_match(
                        t, gt_matches, idx_iou, gt_ignore, ious, idx_det
                    )
                    if m != -1:
                        det_ignore[idx_iou, idx_det] = gt_ignore[m]
                        det_matches[idx_iou, idx_det] = 1
                        gt_matches[idx_iou, m] = 1

        # set unmatched detections outside of area range to ignore
        det_areas = box3d_volume(det)
        det_ignore_area = (det_areas < area_range[0]) | (det_areas > area_range[1])
        ar = det_ignore_area.reshape((1, nb_det))
        det_ignore = torch.logical_or(
            det_ignore,
            torch.logical_and(
                det_matches == 0, torch.repeat_interleave(ar, nb_iou_thrs, 0)
            ),
        )
        det_matches = det_matches.cpu()
        gt_matches = gt_matches.cpu()
        scores_sorted = scores_sorted.cpu()
        gt_ignore = gt_ignore.cpu()
        det_ignore = det_ignore.cpu()

        ret = {
            "dtMatches": det_matches,
            "gtMatches": gt_matches,
            "dtScores": scores_sorted,
            "gtIgnore": gt_ignore,
            "dtIgnore": det_ignore,
        }

        if self.ret_all_prec_rec:
            self.eval_imgs.append(ret)

        return ret

    def _summarize_results(
        self, precisions: torch.Tensor, recalls: torch.Tensor
    ) -> Tuple[MAPMetricResults3D, MARMetricResults]:
        """
        Overloading base class function.
        Summarizes the precision and recall values to calculate mAP/mAR.

        Args:
            precisions:
                Precision values for different thresholds
            recalls:
                Recall values for different thresholds
        """
        results = dict(precision=precisions, recall=recalls)
        map_metrics = MAPMetricResults3D()
        last_max_det_thr = self.max_detection_thresholds[-1]
        map_metrics.map = self._summarize(results, True, max_dets=last_max_det_thr)
        if 0.25 in self.iou_thresholds:
            map_metrics.map_25 = self._summarize(
                results, True, iou_threshold=0.25, max_dets=last_max_det_thr
            )
        if 0.5 in self.iou_thresholds:
            map_metrics.map_50 = self._summarize(
                results, True, iou_threshold=0.5, max_dets=last_max_det_thr
            )

        mar_metrics = MARMetricResults()
        for max_det in self.max_detection_thresholds:
            mar_metrics[f"mar_{max_det}"] = self._summarize(
                results, False, max_dets=max_det
            )

        return map_metrics, mar_metrics

    def compute(self, sem_id_to_name_mapping: Optional[Dict[int, str]] = None) -> dict:
        metrics = MeanAveragePrecision.compute(self)
        final_results = {}

        # resemble class-based results.
        if self.class_metrics:
            seen_classes = self._get_classes()
            if sem_id_to_name_mapping is None:
                logger.warning("No sem_id to name mapping. Falling back on id=name")
                sem_id_to_name_mapping = {
                    sem_id: str(sem_id) for sem_id in seen_classes
                }

            for k, v in metrics.items():
                # Deal with per-class metrics
                if "per_class" in k:
                    # populate per class numbers
                    mapped, unmapped = set(), set()
                    for idx, pcr in enumerate(v):
                        if seen_classes[idx] not in sem_id_to_name_mapping:
                            unmapped.add(seen_classes[idx])
                        else:
                            mapped.add(seen_classes[idx])
                            final_results[
                                f"{k}@{sem_id_to_name_mapping[seen_classes[idx]]}"
                            ] = pcr
                    if len(unmapped) > 0:
                        logger.warning(
                            f"Mapped sem_ids {mapped} but DID NOT MAP sem_ids {unmapped}"
                        )
                else:
                    final_results[k] = v
        else:
            final_results = metrics
        return final_results

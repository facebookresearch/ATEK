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

from typing import List

import numpy as np


def compute_precision_recall_curve(
    y_true: List[bool], y_score: List[float], total_gt_count: int
):
    """Compute precision-recall pairs for different probability thresholds
    This function implementation is based on the scikit-learn
    precision_recall_curve function implementation.
    But it fixes the issue that the scikit-learn function cannot properly
    handle false negatives.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    total_gt_count : int
        Total number of ground-truth samples

    Returns
    -------
    precision : array, shape = [n_thresholds + 1]
        Precision values such that element i is the precision of
        predictions with score >= thresholds[i] and the last element is 1.

    recall : array, shape = [n_thresholds + 1]
        Decreasing recall values such that element i is the recall of
        predictions with score >= thresholds[i] and the last element is 0.

    thresholds : array, shape = [n_thresholds <= len(np.unique(y_score))]
        Increasing thresholds on the decision function used to compute
        precision and recall.
    """

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score should have the same length. ")
    if len(y_true) == 0:
        raise ValueError("y_true or y_score should not be empty.")
    if total_gt_count <= 0:
        raise ValueError("total_gt_count should be positive.")

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = np.cumsum(y_true)[threshold_idxs]
    # false positives with decreasing threshold
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]

    positives = tps + fps
    precision = np.divide(tps, positives, where=(positives != 0))
    recall = tps / total_gt_count

    return precision, recall, thresholds


def compute_average_precision(
    y_true: List[bool], y_score: List[float], total_gt_count: int
):
    """Compute average precision (AP) from prediction scores

    This function implementation is based on the scikit-learn
    precision_recall_curve function implementation.
    But it fixes the issue that the scikit-learn function cannot properly
    handle false negatives.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification

    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function

    total_gt_count : int
        Total number of ground-truth samples

    Returns
    -------
    average_precision : float

    References
    ----------
    .. [1] `Wikipedia entry for the Average precision
           <https://en.wikipedia.org/w/index.php?title=Information_retrieval&
           oldid=793358396#Average_precision>`_
    """
    # precision is sorted in decreasing order, recall is sorted in increasing order
    precision, recall, _ = compute_precision_recall_curve(
        y_true, y_score, total_gt_count
    )
    # Add the precision = 1.0, recall = 0.0 starting point to compute the AP
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]

    # Apply filter to precisions to make sure the precision is no worse than precision
    # value when the recall is higher (otherwise, we can always pick the confidence
    # threshold that achieve both better precision and recall)
    precision = np.maximum.accumulate(precision[::-1])[::-1]

    # Return the step function integral
    return np.sum(np.diff(recall) * np.array(precision)[1:])

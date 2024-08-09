# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from atek.data_preprocess.processors.obb3_gt_processor import Obb3GtProcessor

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EfmGtProcessor(Obb3GtProcessor):
    """
    A Ground truth (GT) processor class for EFM (OBB3 only).
    Child class of `Obb3GtProcessor`, with one more API to return a nested GT dict to contain multi-frame OBB3 GT result.
    """

    def get_gt_by_timestamp_list_ns(self, timestamps_ns: List[int]) -> Optional[Dict]:
        """
        get a GT Dict by timestamps in nanoseconds, returns a nested Dict of the following structure:
            {
                "timestamp_1": {
                    "camera_label_1": {
                        "instance_ids": torch.Tensor (shape: [num_instances], int64)
                        "category_names": list[str],
                        "category_ids": torch.Tensor (shape: [num_instances], int64)
                        "object_dimensions": torch.Tensor (shape: [num_instances, 3], float32, 3 is x, y, z)
                        "ts_world_object": torch.Tensor (shape: [num_instances, 3, 4], float32)
                    },
                    "camera_label_2": {
                        ...
                    }
                    ...
                    }
                    ...
            }
        """
        if len(timestamps_ns) == 0:
            return None

        all_dict = {}

        for single_timestamp in timestamps_ns:
            # call parent class's API to get GT Dict for a single timestamp
            single_dict = self.get_gt_by_timestamp_ns(timestamp_ns=single_timestamp)
            if single_dict is not None:
                all_dict[str(single_timestamp)] = single_dict

        if len(all_dict) == 0:
            return None
        else:
            return all_dict

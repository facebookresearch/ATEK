# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List, Optional, Tuple

import torch


def concat_list_of_tensors(tensor_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Concatenate a list of tensors of (_, 3) into a single tensor of (N, 3), and returns both the stacked tensor and the first dims of the tensors in the list as a
    tensor of (len_of_list).
    """
    lengths_as_tensor = torch.tensor(
        [x.size(0) for x in tensor_list], dtype=torch.int64
    )
    return torch.cat(tensor_list, dim=0), lengths_as_tensor

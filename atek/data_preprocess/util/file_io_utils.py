# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import csv
from typing import Dict, List, Optional, Tuple

import torch


def concat_list_of_tensors(
    tensor_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Concatenate a list of tensors of (_, 3) into a single tensor of (N, 3), and returns both the stacked tensor and the first dims of the tensors in the list as a
    tensor of (len_of_list).
    """
    lengths_of_tensor = torch.tensor(
        [x.size(0) for x in tensor_list], dtype=torch.int64
    )

    if len(tensor_list) > 0:
        return torch.cat(tensor_list, dim=0), lengths_of_tensor
    else:
        return torch.tensor([]), torch.tensor([])


def unpack_list_of_tensors(
    stacked_tensor: torch.Tensor, lengths_of_tensors: torch.Tensor
) -> List[torch.Tensor]:
    """
    Unpack a stacked tensor of (N, 3) back to a list of tensors of (_, 3), according to each subtensor's lengths
    """
    assert lengths_of_tensors.sum().item() == stacked_tensor.size(
        0
    ), "The lengths_of_tensors do not sum to the length of the stacked tensor, {} vs {}".format(
        lengths_of_tensors.sum().item(), stacked_tensor.size(0)
    )

    tensor_list = []
    current_index = 0
    for length in lengths_of_tensors:
        tensor_list.append(stacked_tensor[current_index : current_index + length])
        current_index += length
    return tensor_list


def load_category_mapping_from_csv(
    category_mapping_csv_file: str,
) -> Dict:
    """
    Load the category mapping from a CSV file.

    Args:
        category_mapping_csv_file (str): The path to the category mapping CSV file.
        The CSV file should contain exactly 3 Columns, representing "old_category_name or prototype_name", "atek_category_name", "atek_category_id".

    Returns:
        Dict: The category mapping dictionary in the format of:
            {
                "old_cat/prototype_name"： [“cat_name”, category_id],
                ...
            }
    """
    with open(category_mapping_csv_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert len(header) == 3, "Expected 3 columns in the category mapping csv file"
        assert (
            header[1] == "ATEK Category Name" and header[2] == "ATEK Category Id"
        ), f"Column names must be  ATEK Category Name and ATEK Category Id, but got {header[1]} and {header[2]} instead."
        category_mapping = {rows[0]: (rows[1], rows[2]) for rows in reader}
    return category_mapping

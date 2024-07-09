# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import torch


def fill_or_trim_tensor(tensor: torch.Tensor, dim_size: int, dim: int, fill_value=None):
    """Fill or trim a torch tensor to the given `dim_size`, along the given dim

    Inputs:
        tensor (torch tensor): input tensor
        dim_size (int): the size to fill or trim to (e.g. predefined batch size)
        dim (int): the dimension to fill or trim
        fill_value: if set, fill the tensor with this value. Otherwise, fill by repeating the last element.

    Returns:
        new_tensor (a torch tensor): output tensor with the dim size = `dim_size`.
    """
    assert tensor.shape[dim] > 0, "Input tensor must have at least 1 element"

    original_dim_size = tensor.shape[dim]
    # Trim
    if original_dim_size > dim_size:
        indices = torch.arange(dim_size)
        new_tensor = torch.index_select(tensor, dim, indices)
    # or Fill
    elif original_dim_size < dim_size:
        # fill with given value
        if fill_value is not None:
            fill_shape = list(tensor.shape)
            fill_shape[dim] = dim_size - original_dim_size
            fill = torch.full(fill_shape, fill_value)
        # or repeat last element
        else:
            shape = [1 for _ in range(tensor.ndim)]
            indices = torch.ones(shape).long()
            indices[0] = original_dim_size - 1
            last = torch.take_along_dim(tensor, indices, dim)

            repeat_shape = shape
            repeat_shape[dim] = dim_size - original_dim_size
            fill = last.repeat(repeat_shape)

        new_tensor = torch.cat([tensor, fill], dim=dim)
    else:
        new_tensor = tensor
    return new_tensor


def check_dicts_same_w_tensors(dict_1, dict_2, atol=1e-5) -> bool:
    """
    A utility function to check if two dicts are the same, with special handling of tensors.
    """
    if dict_1.keys() != dict_2.keys():
        return False

    for key in dict_1.keys():
        value1, value2 = dict_1[key], dict_2[key]

        if isinstance(value1, torch.Tensor) and isinstance(value2, torch.Tensor):
            if not torch.allclose(value1, value2, atol=atol):
                return False

        elif isinstance(value1, dict) and isinstance(value2, dict):
            # recursively check sub-dicts
            if not check_dicts_same_w_tensors(value1, value2, atol=atol):
                return False

        elif value1 != value2:
            return False

    return True

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

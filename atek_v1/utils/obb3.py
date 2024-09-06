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

from typing import List, Optional, Union

import torch

from atek_v1.utils.transform_utils import batch_transform_points, get_cuboid_corners
from pytorch3d.transforms import euler_angles_to_matrix


class Obb3:
    """
    A wrapper class for N 3D oriented bounding boxes (Obb3)
    """

    # size: (N, 3) tensor, object size in xyz order
    size: torch.Tensor
    # t_ref_obj: (N, 3) tensor, translation from object (obj) to reference (ref) coordinate in xyz order
    t_ref_obj: torch.Tensor
    # R_ref_obj: (N, 3, 3) tensor, rotation from object (obj) to reference (ref) coordinate
    R_ref_obj: torch.Tensor
    # instance id: (N,) tensor, unique instance id
    instance_id: torch.Tensor
    # category_id: (N,) tensor, category_id id
    category_id: torch.Tensor
    # score: (N,) tensor, confidence score
    score: torch.Tensor

    def __init__(
        self,
        size: torch.Tensor,
        t_ref_obj: torch.Tensor,
        R_ref_obj: torch.Tensor,
        instance_id: torch.Tensor,
        category_id: torch.Tensor,
        score: Optional[torch.Tensor] = None,
    ):
        assert (
            len(size)
            == len(t_ref_obj)
            == len(R_ref_obj)
            == len(instance_id)
            == len(category_id)
        )
        assert size.ndim == 2 and size.shape[1] == 3
        assert t_ref_obj.ndim == 2 and t_ref_obj.shape[1] == 3
        assert R_ref_obj.ndim == 3 and R_ref_obj.shape[1:] == (3, 3)
        assert instance_id.ndim == 1
        assert category_id.ndim == 1
        assert score is None or (score.ndim == 1 and len(score) == len(size))

        self.size = size
        self.t_ref_obj = t_ref_obj
        self.R_ref_obj = R_ref_obj
        self.instance_id = instance_id
        self.category_id = category_id
        if score is None:
            score = torch.ones((len(self.size),))
        self.score = score

    @property
    def T_ref_obj(self):
        return torch.cat((self.R_ref_obj, self.t_ref_obj.unsqueeze(-1)), dim=-1)

    # Nx8x3 corners in reference coordinate frame, reference can be world or camera
    @property
    def bb3_in_ref_frame(self):
        return batch_transform_points(get_cuboid_corners(self.size / 2), self.T_ref_obj)


def init_obb3(
    size: Union[List, torch.Tensor],
    t_ref_obj: Union[List, torch.Tensor],
    euler_angle: Union[List, torch.Tensor],
    instance_id: Union[List, torch.Tensor],
    category_id: Union[List, torch.Tensor],
    score: Optional[Union[List, torch.Tensor]] = None,
    convention: str = "XYZ",
) -> Obb3:
    """
    Initialize Obb3 instance with size, t_ref_obj, and euler_angle
    """
    if not isinstance(size, torch.Tensor):
        size = torch.Tensor(size)
    if not isinstance(t_ref_obj, torch.Tensor):
        t_ref_obj = torch.Tensor(t_ref_obj)
    if not isinstance(euler_angle, torch.Tensor):
        euler_angle = torch.Tensor(euler_angle)
    if not isinstance(instance_id, torch.Tensor):
        instance_id = torch.tensor(instance_id)
    if not isinstance(category_id, torch.Tensor):
        category_id = torch.tensor(category_id)
    if score is not None and not isinstance(score, torch.Tensor):
        score = torch.Tensor(score)
    R_ref_obj = euler_angles_to_matrix(euler_angle, convention=convention)
    return Obb3(
        size,
        t_ref_obj,
        R_ref_obj,
        instance_id,
        category_id,
        score=score,
    )

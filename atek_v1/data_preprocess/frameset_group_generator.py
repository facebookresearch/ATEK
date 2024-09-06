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

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from atek_v1.data_preprocess.data_schema import FramesetGroup
from atek_v1.data_preprocess.data_utils import (
    check_all_same_member,
    generate_disjoint_colors,
    unify_object_target,
)
from atek_v1.data_preprocess.frameset_aligner import FramesetAligner

# from atek_v1.utils import mesh_boolean_utils

from projectaria_tools.core.sophus import SE3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class FramesetSelectionConfig:
    """
    Config class for selecting framesets. The basic idea to group frameset is based on
    the temporal information combined with the spatial information. For example, the simplest
    way to group frameset is starting from every frameset in order and find n framesets to group
    them, aka, group n framesets starting from first frameset with stride of 1.
    """

    # The number of framesets to group together.
    num_framesets_per_group: int = 3

    # Skip first n framesets in the frameset aligner.
    skip_first_n_framesets: int = 0

    # Skip last n framesets in the frameset aligner.
    skip_last_n_framesets: int = 0

    # The stride between two consecutive framesets group first frameset.
    stride: int = 1

    # Time threshold for selecting framesets. If set to None, we don't do any timestamp
    # thresholding for selection frameset for grouping.
    time_duration_ns_threshold: Optional[int] = None

    # Distance threshold for selecting framesets. If set to None, we don't do any translation
    # thresholding for selection frameset for grouping.
    translation_m_threshold: Optional[float] = None

    # Rotation threshold for selecting framesets. If set to None, we don't do any rotation
    # thresholding for selection frameset for grouping.
    rotation_deg_threshold: Optional[float] = None

    # Fov overlapping ratio threshold for selecting framesets. If set to None, we don't do any
    # overlapping ratio thresholding for selection frameset for grouping. This mechanism is that build
    # a fov for depth fov near clipping distance to fov far clipping distance, and compute
    # the overlapping ratio between two framesets to make sure the viewing perspective have changed enough.
    # We only select a frameset when the overlapping ratio is getting smaller than this threshold.
    # CAUTION: The compute of the mesh intersection is expensive and slow so only use this when necessary.
    fov_overlapping_ratio_threshold: Optional[float] = None

    # The near clipping distance for computing the fov overlapping ratio.
    far_clipping_distance: Optional[float] = 4.0

    # The frameset idx which selected as the local coordinate frame.
    local_selection: int = 0

    def __str__(self):
        return "FramesetSelectionConfig:\n" + "\n".join(
            [f"  {k}: {v}" for k, v in asdict(self).items()]
        )


class FramesetGroupGenerator:
    def __init__(
        self,
        frameset_aligner: FramesetAligner,
        frameset_selection_config: FramesetSelectionConfig,
        require_objects: bool = False,
    ):
        self.frameset_aligner = frameset_aligner
        self.frameset_selection_config = frameset_selection_config
        self.require_objects = require_objects

        self.sanity_check_inputs()
        self.frameset_ids_for_groups = self.process_framesets()

    def sanity_check_inputs(self):
        """
        Sanity check frameset grouping settings and inputs.
        """
        if (
            (self.frameset_selection_config.translation_m_threshold is not None)
            or (self.frameset_selection_config.rotation_deg_threshold is not None)
            or (
                self.frameset_selection_config.fov_overlapping_ratio_threshold
                is not None
            )
        ):
            assert (
                self.frameset_aligner.mps_data_processor is not None
            ), "Require trajectory information for frameset selection."

        assert self.frameset_selection_config.stride >= 1

    def process_framesets(self):
        """
        Process the framesets for grouping.
        """
        if self.frameset_selection_config.fov_overlapping_ratio_threshold is not None:
            self.frameset_aligner.update_frameset_fov_mesh(
                far_clipping_distance=self.frameset_selection_config.far_clipping_distance
            )

        num_framesets = self.frameset_aligner.aligned_frameset_number()
        frameset_ids_for_groups = []

        logger.info(
            f"Start grouping framesets based on the settings:\n {self.frameset_selection_config}"
        )

        for i in range(
            self.frameset_selection_config.skip_first_n_framesets,
            num_framesets - self.frameset_selection_config.skip_last_n_framesets,
            self.frameset_selection_config.stride,
        ):
            frameset_group_ids = self.group_framesets_by_setting(i)
            if frameset_group_ids is not None:
                frameset_ids_for_groups.append(frameset_group_ids)

        logger.info(
            f"Framesets grouping finished with {len(frameset_ids_for_groups)} "
            f"groups from {num_framesets} aligned framesets."
        )

        return frameset_ids_for_groups

    def get_frameset_selection_info_by_index(self, index: int):
        """
        Get the selected frameset info by index.
        """
        info = {}
        if self.frameset_selection_config.time_duration_ns_threshold is not None:
            info["frameset_timestamp_ns"] = (
                self.frameset_aligner.get_frameset_timestamp_by_index(index)
            )

        if (self.frameset_selection_config.translation_m_threshold is not None) or (
            self.frameset_selection_config.rotation_deg_threshold is not None
        ):
            info["T_world_frameset"] = (
                self.frameset_aligner.get_T_world_frameset_by_index(index)
            )

        if self.frameset_selection_config.fov_overlapping_ratio_threshold is not None:
            frameset_fov_in_world = (
                self.frameset_aligner.get_frameset_fov_mesh()
            ).apply_transform(
                self.frameset_aligner.get_T_world_frameset_by_index(index).to_matrix()
            )
            info["frameset_fov_in_world"] = frameset_fov_in_world

        return info

    def group_framesets_by_setting(
        self,
        start_index: int,
    ):
        """
        Group framesets by setting starting from the start_index frameset.
        """
        num_framesets = (
            self.frameset_aligner.aligned_frameset_number()
            - self.frameset_selection_config.skip_last_n_framesets
        )
        assert (
            0 <= start_index < num_framesets
        ), f"Start index {start_index} | {num_framesets}"

        if (
            (self.frameset_selection_config.time_duration_ns_threshold is None)
            and (self.frameset_selection_config.translation_m_threshold is None)
            and (self.frameset_selection_config.rotation_deg_threshold is None)
            and (self.frameset_selection_config.fov_overlapping_ratio_threshold is None)
        ):
            # Just try to get the consecutive framesets if we don't need to check any
            # grouping criteria.

            # Quick return if not enough framesets left to group.
            if (
                num_framesets - start_index - 1
                < self.frameset_selection_config.num_framesets_per_group - 1
            ):
                return None
            else:
                return list(
                    range(
                        start_index,
                        start_index
                        + self.frameset_selection_config.num_framesets_per_group,
                    )
                )

        frameset_group_ids = [start_index]
        last_frameset_info = self.get_frameset_selection_info_by_index(start_index)
        for i in range(start_index + 1, num_framesets):
            if (
                len(frameset_group_ids)
                == self.frameset_selection_config.num_framesets_per_group
            ):
                return frameset_group_ids

            # Quick return if not enough framesets left to group.
            if (
                num_framesets - i
                < self.frameset_selection_config.num_framesets_per_group
                - len(frameset_group_ids)
            ):
                return None

            current_frameset_info = self.get_frameset_selection_info_by_index(i)
            if self.frameset_selection_config.time_duration_ns_threshold is not None:
                if (
                    current_frameset_info["frameset_timestamp_ns"]
                    - last_frameset_info["frameset_timestamp_ns"]
                ) >= self.frameset_selection_config.time_duration_ns_threshold:
                    frameset_group_ids.append(i)
                    last_frameset_info = current_frameset_info
                    continue

            if self.frameset_selection_config.translation_m_threshold is not None:
                distance = np.linalg.norm(
                    current_frameset_info["T_world_frameset"].translation()
                    - last_frameset_info["T_world_frameset"].translation()
                )
                if distance >= self.frameset_selection_config.translation_m_threshold:
                    frameset_group_ids.append(i)
                    last_frameset_info = current_frameset_info
                    continue

            if self.frameset_selection_config.rotation_deg_threshold is not None:
                R = (
                    last_frameset_info["T_world_frameset"].rotation().inverse()
                    @ current_frameset_info["T_world_frameset"].rotation()
                ).to_matrix()
                trace = np.trace(R)
                theta = np.arccos((trace - 1) / 2) * 180 / np.pi
                if theta >= self.frameset_selection_config.rotation_deg_threshold:
                    frameset_group_ids.append(i)
                    last_frameset_info = current_frameset_info
                    continue

            if (
                self.frameset_selection_config.fov_overlapping_ratio_threshold
                is not None
            ):
                # TODO: temporarily disabled to pass CI. The preprocessing part of atek_v1 should NOT be used!
                """
                fov_intersection = mesh_boolean_utils.intersect_meshes(
                    last_frameset_info["frameset_fov_in_world"],
                    current_frameset_info["frameset_fov_in_world"],
                )
                if (
                    fov_intersection.volume
                    / last_frameset_info["frameset_fov_in_world"].volume
                    <= self.frameset_selection_config.fov_overlapping_ratio_threshold
                ):
                    frameset_group_ids.append(i)
                    last_frameset_info = current_frameset_info
                    continue
                """
                pass

        return None

    def frameset_group_number(self):
        return len(self.frameset_ids_for_groups)

    def get_frameset_group_by_index(self, index: int) -> FramesetGroup:
        assert 0 <= index < self.frameset_group_number(), f"Index {index} out of bound."
        frameset_group = FramesetGroup()
        frameset_group.framesets = [
            self.frameset_aligner.get_frameset_by_index(frameset_id)
            for frameset_id in self.frameset_ids_for_groups[index]
        ]

        # Fill in the frameset and some sanity checks
        assert (
            len(frameset_group.framesets) > 0
        ), "No valid framesets found for frameset group."
        assert check_all_same_member(
            frameset_group.framesets, "data_source"
        ), "All the frames must be from the same data source."
        check_all_same_member(
            frameset_group.framesets, "sequence_name"
        ), "All the frames must be from the same sequence."

        frameset_group.data_source = frameset_group.framesets[0].data_source
        frameset_group.sequence_name = frameset_group.framesets[0].sequence_name
        frameset_group.local_selection = self.frameset_selection_config.local_selection

        # Generate trajectory information if available.
        if frameset_group.framesets[0].T_world_frameset is not None:
            Ts_world_frameset = [
                SE3.from_matrix3x4(frameset.T_world_frameset)
                for frameset in frameset_group.framesets
            ]
            T_world_local = Ts_world_frameset[
                self.frameset_selection_config.local_selection
            ]
            T_local_world = T_world_local.inverse()
            frameset_group.Ts_local_frameset = [
                (T_local_world @ T_world_frameset).to_matrix3x4()
                for T_world_frameset in Ts_world_frameset
            ]
            frameset_group.T_world_local = T_world_local.to_matrix3x4()

            frameset_group.gravity_in_world = np.column_stack(
                [frameset.gravity_in_world for frameset in frameset_group.framesets]
            ).mean(axis=1, keepdims=True)
            frameset_group.gravity_in_local = (
                T_local_world.rotation() @ frameset_group.gravity_in_world
            )
            if self.require_objects:
                unify_object_target(frameset_group)
                frameset_group.Ts_local_object = [
                    (T_local_world @ SE3.from_matrix3x4(T_world_object)).to_matrix3x4()
                    for T_world_object in frameset_group.Ts_world_object
                ]

        return frameset_group

    def get_frameset_group_fov_mesh_by_index(self, index: int):
        """
        Return frameset group FOV mesh in the world coordinate.
        """
        assert 0 <= index < self.frameset_group_number(), f"Index {index} out of bound."
        frameset_meshes = []
        colors = generate_disjoint_colors(
            self.frameset_selection_config.num_framesets_per_group
        )
        for i, frameset_id in enumerate(self.frameset_ids_for_groups[index]):
            frameset_mesh = self.frameset_aligner.get_frameset_fov_mesh()
            T_world_frameset = self.frameset_aligner.get_T_world_frameset_by_index(
                frameset_id
            )
            frameset_mesh.apply_transform(T_world_frameset.to_matrix())
            frameset_mesh.visual.vertex_colors = colors[i]
            frameset_meshes.append(frameset_mesh)
        return sum(frameset_meshes)

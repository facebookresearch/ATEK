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

import re
from dataclasses import dataclass, fields
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class Frame:
    """
    A class to store the information of a frame in a video sequence.
    """

    # ------------------------ ARIA IMAGE DATA BEGIN -------------------------
    # The index of the frame in the vrs.
    frame_id: int = None

    # The sequence name of the vrs.
    sequence_name: str = None

    # The dataset name
    data_source: str = None

    # Camera model. e.g. [Pinhole, Fisheye, etc]
    camera_model: str = None

    # Camera model intrinsics parameters.
    camera_parameters: np.ndarray = None

    # Stream id name.
    stream_id: str = None

    # Camera label name.
    camera_name: str = None

    # Timestamp of the frame.
    timestamp_ns: int = None

    # Image data in rgb order in numpy array. HxWxC
    image: np.ndarray = None
    # ------------------------ ARIA IMAGE DATA END ---------------------------

    # +++++++++++++++++++++++ MPS DATA BEGIN +++++++++++++++++++++++++++
    # Gravity direction in camera coordinate frame. 3x1
    gravity_in_camera: np.ndarray = None

    # Gravity direction in world coordinate frame. 3x1
    gravity_in_world: np.ndarray = None

    # Transformation from camera to world coordinate frame. 3x4 [R|t]
    T_world_camera: np.ndarray = None

    # +++++++++++++++++++++++ MPS DATA END +++++++++++++++++++++++++++++

    # ======================= GROUND-TRUTH DATA BEGIN ===============================
    # Mapping for category id to name.
    category_id_to_name: Dict[int, str] = None

    # Object instances ids.
    object_instance_ids: List[int] = None

    # Object category semantic ids. Make sure this is 1 to 1 aligned with the instance ids.
    object_category_ids: List[int] = None

    # Transformation form object to camera for each object instance. List[ 3x4 [R|t] ]
    Ts_camera_object: List[np.ndarray] = None

    # Transformation from object to world for each object instance. List[ 3x4 [R|t] ]
    Ts_world_object: List[np.ndarray] = None

    # Object bounding box dimensions in meters. List[ 3 [xyz] ]
    object_dimensions: List[np.ndarray] = None

    # Object 2d bounding boxes in pixels. List[ [xmin, xmax, ymin, ymax] ]
    bb2ds: List[np.ndarray] = None
    # ======================= TARGET DATA END =================================

    @staticmethod
    def image_fields():
        return [
            "image",
        ]

    @staticmethod
    def camera_info_fields():
        return [
            "camera_model",
            "camera_name",
        ]

    @staticmethod
    def camera_data_fields():
        return [
            "camera_parameters",
        ]

    @staticmethod
    def input_info_fields():
        return [
            "frame_id",
            "sequence_name",
            "data_source",
            "stream_id",
            "timestamp_ns",
        ]

    @staticmethod
    def trajectory_data_fields():
        return ["gravity_in_camera", "gravity_in_world", "T_world_camera"]

    @staticmethod
    def object_info_fields():
        return [
            "category_id_to_name",
            "object_instance_ids",
            "object_category_ids",
        ]

    @staticmethod
    def object_2d_data_fields():
        return [
            "bb2ds",
        ]

    @staticmethod
    def object_3d_data_fields():
        return [
            "Ts_camera_object",
            "Ts_world_object",
            "object_dimensions",
        ]

    @staticmethod
    def flatten_dict_key_pattern():
        return re.compile(r"F#\d+-\d+\+.+")

    def to_flatten_dict(self):
        """
        Transforms the frame attributes into a flattened dictionary, excluding
        attributes with None values. Attributes are prefixed to ensure uniqueness and
        to maintain context.
        """
        flatten_dict = {
            f"F#{self.stream_id}+{f.name}": getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }
        return flatten_dict


@dataclass
class Frameset:
    """
    A class to store the information of a set of frames in a video sequence.
    """

    # ------------------------ INPUT IMAGE DATA BEGIN -------------------------
    # The frames in the frame set.
    frames: List[Frame] = None

    # The sequence name of the vrs.
    sequence_name: str = None

    # The dataset name
    data_source: str = None

    # Timestamp of the frameset which could be configured to different settings.
    # For example, it could be the average timestamp of the frames' time stamp.
    # It could also be the timestamp of the first frame in the frameset. Could
    # be extended base on the needs.
    timestamp_ns: int = None

    # Timestamp type. [Average, Earliest, ...]
    timestamp_type: str = None

    # Frameset rig origin selection method. Different ways to do that, could be set to
    # one camera, some virtual coordinate, or some other methods. Could be extended base
    # on the needs. [camera_name, stream_name, etc]
    origin_selection: str = None

    # Transformation from frameset to camera coordinate frame. 3x4 [R|t]
    # Note this is aligned with the frame list.
    Ts_frameset_camera: List[np.ndarray] = None
    # ------------------------ INPUT IMAGE DATA END ---------------------------

    # +++++++++++++++++++++++ MPS DATA BEGIN +++++++++++++++++++++++++++

    # Gravity direction in frameset coordinate frame. 3x1
    gravity_in_frameset: np.ndarray = None

    # Gravity direction in world coordinate frame. 3x1
    gravity_in_world: np.ndarray = None

    # Transformation from frameset to world coordinate frame. 3x4 [R|t]
    T_world_frameset: np.ndarray = None

    # Semi-dense point cloud, points in world coordinate frame. Nx3
    points_world: Optional[torch.Tensor] = None

    # Semi-dense point cloud, inverse points distance std, Nx1
    points_inv_dist_std: Optional[torch.Tensor] = None

    # +++++++++++++++++++++++ MPS DATA END +++++++++++++++++++++++++++++

    # ======================= TARGET DATA BEGIN ===============================
    # Frameset target is in the context for the frameset. For example, all the objects
    # seen by any frame in the frameset are considered as the target of the frameset.

    # Mapping for category id to name.
    category_id_to_name: Dict[int, str] = None

    # Object instances ids.
    object_instance_ids: List[int] = None

    # Object category semantic ids. Make sure this is 1 to 1 aligned with the instance ids.
    object_category_ids: List[int] = None

    # Transformation from object to world for each object instance. List[ 3x4 [R|t] ]
    Ts_world_object: List[np.ndarray] = None

    # Object bounding box dimensions in meters. List[ 3 [xyz] ]
    object_dimensions: List[np.ndarray] = None

    # Transformation form object to frame for each object instance. List[ 3x4 [R|t] ]
    Ts_frameset_object: List[np.ndarray] = None

    # ======================= TARGET DATA END =================================
    @staticmethod
    def input_info_fields():
        return [
            "sequence_name",
            "data_source",
            "timestamp_ns",
            "timestamp_type",
            "origin_selection",
        ]

    @staticmethod
    def camera_data_fields():
        return [
            "Ts_frameset_camera",
        ]

    @staticmethod
    def trajectory_data_fields():
        return ["gravity_in_frameset", "gravity_in_world", "T_world_frameset"]

    @staticmethod
    def semidense_points_fields():
        return [
            "points_world",
            "points_inv_dist_std",
        ]

    @staticmethod
    def object_info_fields():
        return [
            "category_id_to_name",
            "object_instance_ids",
            "object_category_ids",
        ]

    @staticmethod
    def object_3d_data_fields():
        return [
            "Ts_frameset_object",
            "Ts_world_object",
            "object_dimensions",
        ]

    @staticmethod
    def flatten_dict_key_pattern():
        return re.compile(r"FS\+.+")

    def to_flatten_dict(self):
        """
        Transforms the frameset's attributes into a flattened dictionary, excluding
        attributes with None values. The nested Frame structure is not preserved in
        the resulting dictionary. Attributes are prefixed to ensure uniqueness and
        to maintain context.
        """
        flatten_dict = {}
        for frame in self.frames:
            flatten_dict.update(frame.to_flatten_dict())

        for f in fields(self):
            if f.name == "frames":
                continue
            if getattr(self, f.name) is not None:
                flatten_dict[f"FS+{f.name}"] = getattr(self, f.name)

        return flatten_dict


@dataclass
class FramesetGroup:
    """
    A class to store the information of a group of framesets in a video sequence.
    """

    # ------------------------ INPUT IMAGE DATA BEGIN -------------------------
    # The framesets in the frameset group.
    framesets: List[Frameset] = None

    # The sequence name of the vrs.
    sequence_name: str = None

    # The dataset name
    data_source: str = None

    # Local coordinate frame selection method. Pick the ith frameset rig coordinate
    # as the local coordinate frame.
    local_selection: int = None
    # ------------------------ INPUT IMAGE DATA END ---------------------------

    # +++++++++++++++++++++++ TRAJECTORY DATA BEGIN +++++++++++++++++++++++++++
    # Gravity direction in frameset coordinate frame. 3x1
    gravity_in_local: np.ndarray = None

    # Gravity direction in world coordinate frame. 3x1
    gravity_in_world: np.ndarray = None

    # Transformation from local to world coordinate frames. 3x4 [R|t]
    T_world_local: np.ndarray = None

    # Transformation from local to frameset coordinate frames. 3x4 [R|t]
    # Note this is aligned with the frameset list.
    Ts_local_frameset: List[np.ndarray] = None
    # +++++++++++++++++++++++ TRAJECTORY DATA END +++++++++++++++++++++++++++++

    # ======================= TARGET DATA BEGIN ===============================
    # Frameset target is in the context for the frameset. For example, all the objects
    # seen by any frame in the frameset are considered as the target of the frameset.

    # Mapping for category id to name.
    category_id_to_name: Dict[int, str] = None

    # Object instances ids.
    object_instance_ids: List[int] = None

    # Object category semantic ids. Make sure this is 1 to 1 aligned with the instance ids.
    object_category_ids: List[int] = None

    # Transformation from object to world for each object instance. List[ 3x4 [R|t] ]
    Ts_world_object: List[np.ndarray] = None

    # Object bounding box dimensions in meters. List[ 3 [xyz] ]
    object_dimensions: List[np.ndarray] = None

    # Transformation form object to frame for each object instance. List[ 3x4 [R|t] ]
    Ts_local_object: List[np.ndarray] = None
    # ======================= TARGET DATA END =================================

    @staticmethod
    def input_info_fields():
        return [
            "sequence_name",
            "data_source",
            "local_selection",
        ]

    @staticmethod
    def trajectory_data_fields():
        return [
            "gravity_in_local",
            "gravity_in_world",
            "T_world_local",
            "Ts_local_frameset",
        ]

    @staticmethod
    def object_info_fields():
        return [
            "category_id_to_name",
            "object_instance_ids",
            "object_category_ids",
        ]

    @staticmethod
    def object_3d_data_fields():
        return [
            "Ts_local_object",
            "Ts_world_object",
            "object_dimensions",
        ]

    @staticmethod
    def flatten_dict_key_pattern():
        return re.compile(r"FSG\+.+")

    def to_flatten_dict(self):
        """
        Transforms the FramesetGroup's attributes into a flattened dictionary, excluding
        attributes with None values. The nested Frame/Frameset structure is not preserved in
        the resulting dictionary. Attributes are prefixed to ensure uniqueness and
        to maintain context.
        """
        flatten_dict = {}
        for i, frameset in enumerate(self.framesets):
            frameset_flatten_dict = frameset.to_flatten_dict()
            for k, v in frameset_flatten_dict.items():
                if i == 0:
                    flatten_dict[k] = [v]
                else:
                    flatten_dict[k].append(v)

        for f in fields(self):
            if f.name == "framesets":
                continue
            if getattr(self, f.name) is not None:
                flatten_dict[f"FSG+{f.name}"] = getattr(self, f.name)

        return flatten_dict

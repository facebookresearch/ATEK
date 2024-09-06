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

import json
import os
from dataclasses import asdict
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from atek_v1.data_preprocess.data_schema import Frame
from atek_v1.dataset.atek_raw_dataset import AriaStreamIds, AtekRawFrameDataset
from atek_v1.dataset.atek_webdataset import create_atek_webdataset
from atek_v1.utils.camera_utils import linear_cam_matrix_from_intrinsics

from detectron2.structures import Boxes, Instances
from webdataset.filters import pipelinefilter
from webdataset.shardlists import single_node_only

KEY_MAPPING = {
    "f#214-1+image": "images",
    "F#214-1+camera_parameters": "camera_params",
    "F#214-1+Ts_camera_object": "Ts_camera_object",
    "F#214-1+object_dimensions": "object_dimensions",
    "F#214-1+bb2ds": "bb2ds_x0x1y0y1",
    "F#214-1+category_id_to_name": "category_id_to_name",
    "F#214-1+object_category_ids": "object_category_ids",
    "F#214-1+object_instance_ids": "object_instance_ids",
    "F#214-1+sequence_name": "sequence_name",
    "F#214-1+frame_id": "frame_id",
    "F#214-1+timestamp_ns": "timestamp_ns",
    "F#214-1+T_world_camera": "T_world_camera",
    "F#214-1+Ts_world_object": "Ts_world_object",
    "F#214-1+object_dimensions": "object_dimensions",
}


class ObjectDetectionMode(Enum):
    """
    An enum class to indicate if detection is per-category, or per instance:
    - PER_CATEGORY: will use the `category_id` (or `category_name`) field as the output label.
    - PER_INSTANCE: will use the `instance_id` field as the output label.
    """

    PER_CATEGORY = 1
    PER_INSTANCE = 2


def omni3d_key_selection(key: str) -> bool:
    return key in KEY_MAPPING.keys()


def omni3d_key_remap(key: str) -> str:
    if key in KEY_MAPPING.keys():
        return KEY_MAPPING[key]
    else:
        return key


def get_id_map(id_map_json):
    with open(id_map_json, "r") as f:
        id_map = json.load(f)
        id_map = {k: int(v) for k, v in id_map.items()}
    return id_map


def atek_to_omni3d(
    data,
    category_id_remapping: Optional[Dict] = None,
    object_detection_mode: ObjectDetectionMode = ObjectDetectionMode.PER_CATEGORY,
    min_bb2d_area=100,
    min_bb3d_depth=0.3,
    max_bb3d_depth=5,
):
    """
    A helper data transform function to convert the ATEK webdataset data to omni3d unbatched
    samples. Yield one unbatched sample a time to use the collation and batching mechanism in
    the webdataset properly.
    """
    for sample in data:
        # Image rgb->bgr->255->uint8
        assert sample["images"].shape[1] == 3 and sample["images"].ndim == 4
        images = (sample["images"].flip(1) * 255).to(torch.uint8)

        # Image information
        Ks = linear_cam_matrix_from_intrinsics(sample["camera_params"])
        image_height, image_width = sample["images"].shape[2:]

        for idx in range(len(images)):
            sample_new = {
                "image": images[idx],
                "K": Ks[idx].tolist(),  # CubeRCNN requires list input
                "height": image_height,
                "width": image_width,
                # Metadata
                "frame_id": sample["frame_id"][idx],
                "timestamp_ns": sample["timestamp_ns"][idx],
                "sequence_name": os.path.basename(
                    os.path.dirname(sample["sequence_name"][idx])
                ),
            }

            # Populate instances for omni3d
            instances = Instances(images.shape[2:])
            if sample["bb2ds_x0x1y0y1"][idx] is not None:
                bb2ds_x0y0x1y1 = sample["bb2ds_x0x1y0y1"][idx][:, [0, 2, 1, 3]]

                bb2ds_area = (bb2ds_x0y0x1y1[:, 2] - bb2ds_x0y0x1y1[:, 0]) * (
                    bb2ds_x0y0x1y1[:, 3] - bb2ds_x0y0x1y1[:, 1]
                )
                filter_bb2ds_area = bb2ds_area > min_bb2d_area

                bb3ds_depth = sample["Ts_camera_object"][idx][:, -1, -1]
                filter_bb3ds_depth = (min_bb3d_depth <= bb3ds_depth) & (
                    bb3ds_depth <= max_bb3d_depth
                )

                # Remap the semantic id and filter classes
                if object_detection_mode == ObjectDetectionMode.PER_CATEGORY:
                    # Use category names as category ids for category-based detection
                    sem_ids = [
                        sample["category_id_to_name"][0][str(cat_id)]
                        for cat_id in sample["object_category_ids"][0]
                    ]
                elif object_detection_mode == ObjectDetectionMode.PER_INSTANCE:
                    # Use instance ids as category ids for instance-based detection
                    sem_ids = [
                        str(inst_id) for inst_id in sample["object_instance_ids"][idx]
                    ]
                else:
                    raise ValueError(
                        f"Unsupported object detection mode: {object_detection_mode}"
                    )

                # Remap to Omni3D semantic ids, which should be from 0 to N.
                if category_id_remapping is not None:
                    sem_ids = [category_id_remapping.get(id, -1) for id in sem_ids]
                sem_ids = torch.tensor(sem_ids)
                filter_ids = sem_ids >= 0

                final_filter = filter_bb2ds_area & filter_bb3ds_depth & filter_ids
                Ts_camera_object_filtered = sample["Ts_camera_object"][idx][
                    final_filter
                ]
                ts_camera_object_filtered = Ts_camera_object_filtered[:, :, 3]

                filtered_projection_2d = (
                    Ks[idx].repeat(len(ts_camera_object_filtered), 1, 1)
                    @ ts_camera_object_filtered.unsqueeze(-1)
                ).squeeze(-1)
                filtered_projection_2d = filtered_projection_2d[
                    :, :2
                ] / filtered_projection_2d[:, 2].unsqueeze(-1)

                instances.gt_classes = sem_ids[final_filter]
                instances.gt_boxes = Boxes(bb2ds_x0y0x1y1[final_filter])
                instances.gt_poses = Ts_camera_object_filtered[:, :, :3]
                instances.gt_boxes3D = torch.cat(
                    [
                        filtered_projection_2d,
                        bb3ds_depth[final_filter].unsqueeze(-1),
                        # Omni3d has the inverted zyx dimensions
                        # https://github.com/facebookresearch/omni3d/blob/main/cubercnn/util/math_util.py#L144C1-L181C40
                        sample["object_dimensions"][idx][final_filter].flip(-1),
                        ts_camera_object_filtered,
                    ],
                    axis=-1,
                )

                sample_new["instances"] = instances
                filtered_category_ids = [
                    str(cat_id)
                    for cat_id, flag in zip(
                        sample["object_category_ids"][idx], final_filter
                    )
                    if flag
                ]
                category = [
                    sample["category_id_to_name"][idx][cat_id]
                    for cat_id in filtered_category_ids
                ]
                sample_new.update(
                    {
                        "T_world_camera": sample["T_world_camera"][idx],
                        "Ts_world_object": sample["Ts_world_object"][idx][final_filter],
                        "object_dimensions": sample["object_dimensions"][idx][
                            final_filter
                        ],
                        "bb2ds_x0x1y0y1": sample["bb2ds_x0x1y0y1"][idx][final_filter],
                        "category": category,
                    }
                )
                yield sample_new


def collate_as_list(batch):
    return list(batch)


def create_omni3d_webdataset(
    urls: List,
    batch_size: int,
    repeat: bool = False,
    nodesplitter: Callable = single_node_only,
    category_id_remapping_json: Optional[str] = None,
    object_detection_mode: ObjectDetectionMode = ObjectDetectionMode.PER_CATEGORY,
    min_bb2d_area=100,
    min_bb3d_depth=0.3,
    max_bb3d_depth=5,
):
    if category_id_remapping_json is not None:
        category_id_remapping = get_id_map(category_id_remapping_json)
    else:
        category_id_remapping = None

    return create_atek_webdataset(
        urls,
        batch_size,
        nodesplitter=nodesplitter,
        select_key_fn=omni3d_key_selection,
        remap_key_fn=omni3d_key_remap,
        data_transform_fn=pipelinefilter(atek_to_omni3d)(
            category_id_remapping,
            object_detection_mode,
            min_bb2d_area,
            min_bb3d_depth,
            max_bb3d_depth,
        ),
        collation_fn=collate_as_list,
        repeat=repeat,
    )


def atek_raw_to_omni3d(rgb_image_frame: Frame) -> List[Dict]:
    K = linear_cam_matrix_from_intrinsics(
        rgb_image_frame.camera_parameters[np.newaxis, ...]
    )[0]
    image = rgb_image_frame.image
    # image color RGB to BGR
    image = image[:, :, [2, 1, 0]]

    # image dimension (height, width, channel) to (channel, height, width)
    image = image.transpose(2, 0, 1)

    frame = {
        "data_source": rgb_image_frame.data_source,
        "sequence_name": rgb_image_frame.sequence_name.split("/")[-3],
        "frame_id": rgb_image_frame.frame_id,
        "timestamp_ns": rgb_image_frame.timestamp_ns,
        "T_world_camera": rgb_image_frame.T_world_camera,
        "image": torch.as_tensor(np.ascontiguousarray(image)),
        "height": image.shape[1],
        "width": image.shape[2],
        "K": K,
    }

    # GT object annotation info
    if "Ts_world_object" in asdict(rgb_image_frame).keys():
        frame["Ts_world_object"] = rgb_image_frame.Ts_world_object
        frame["object_dimensions"] = rgb_image_frame.object_dimensions
        frame["category"] = [
            rgb_image_frame.category_id_to_name[cat_id]
            for cat_id in rgb_image_frame.object_category_ids
        ]
        frame["bb2ds_x0x1y0y1"] = rgb_image_frame.bb2ds

    return [frame]


def create_omni3d_raw_dataset(
    raw_data_path: str,
    selected_device_number: int = 0,
    rotate_image_cw90deg: bool = True,
    target_image_resolution: Tuple[int, int] = (512, 512),
):
    atek_omni3d_raw_dataset = AtekRawFrameDataset(
        raw_data_path,
        selected_device_number=selected_device_number,
        stream_id=AriaStreamIds.rgb_stream_id,
        rotate_image_cw90deg=rotate_image_cw90deg,
        target_image_resolution=target_image_resolution,
        transform_fn=atek_raw_to_omni3d,
    )
    return atek_omni3d_raw_dataset

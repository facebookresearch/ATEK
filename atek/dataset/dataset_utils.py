# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
from typing import List


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# STREAM_IDS = ["214-1", "1201-1", "1201-2"]
STREAM_IDS = ["214-1"]

SELECTED_KEYS = []
KEY_MAPPING = {
    "f#214-1+image_0.jpeg": "image",
    "F#214-1+camera_parameters": "camera_params",
    "F#214-1+Ts_camera_object": "Ts_camera_object",
    "F#214-1+object_dimensions": "original_object_dimensions",
    "F#214-1+bb2ds": "bbox_2d",
    "F#214-1+category_id_to_name": "category_id_to_name",
    "F#214-1+object_category_ids": "object_category_ids",
    "F#214-1+object_instance_ids": "object_instance_ids",
    "F#214-1+sequence_name": "sequence_name",
    "F#214-1+frame_id": "frame_id",
    "F#214-1+timestamp_ns": "timestamp_ns",
}

IMAGE_KEYS = []
TENSOR_KEYS = []
INFO_KEYS = []

for stream_id in STREAM_IDS:
    IMAGE_KEYS.append(f"f#{stream_id}+image_0.jpeg")

    TENSOR_KEYS.append(f"F#{stream_id}+camera_parameters")
    TENSOR_KEYS.append(f"F#{stream_id}+camera_model")
    TENSOR_KEYS.append(f"F#{stream_id}+camera_name")

    TENSOR_KEYS.append(f"F#{stream_id}+bb2ds")
    TENSOR_KEYS.append(f"F#{stream_id}+T_world_camera")
    TENSOR_KEYS.append(f"F#{stream_id}+Ts_world_object")
    TENSOR_KEYS.append(f"F#{stream_id}+Ts_camera_object")
    TENSOR_KEYS.append(f"F#{stream_id}+object_dimensions")

    INFO_KEYS.append(f"F#{stream_id}+sequence_name")
    INFO_KEYS.append(f"F#{stream_id}+frame_id")
    INFO_KEYS.append(f"F#{stream_id}+timestamp_ns")

    INFO_KEYS.append(f"F#{stream_id}+category_id_to_name")
    INFO_KEYS.append(f"F#{stream_id}+object_instance_ids")
    INFO_KEYS.append(f"F#{stream_id}+object_category_ids")

# TENSOR_KEYS.append("FS+T_world_frameset")
# TENSOR_KEYS.append("FS+Ts_frameset_camera")
# TENSOR_KEYS.append("FS+Ts_world_object")
# TENSOR_KEYS.append("FS+Ts_frameset_object")
# TENSOR_KEYS.append("FS+object_dimensions")

# TENSOR_KEYS.append("FSG+T_world_local")
# TENSOR_KEYS.append("FSG+Ts_local_frameset")
# TENSOR_KEYS.append("FSG+Ts_world_object")
# TENSOR_KEYS.append("FSG+Ts_local_object")
# TENSOR_KEYS.append("FSG+object_dimensions")

# INFO_KEYS.append("FS+origin_selection")

# INFO_KEYS.append("FS+category_id_to_name")
# INFO_KEYS.append("FS+object_instance_ids")
# INFO_KEYS.append("FS+object_category_ids")

# INFO_KEYS.append("FSG+category_id_to_name")
# INFO_KEYS.append("FSG+object_instance_ids")
# INFO_KEYS.append("FSG+object_category_ids")


SELECTED_KEYS = IMAGE_KEYS + TENSOR_KEYS + INFO_KEYS


def get_rank_world_size(group=None):
    """Return node and worker info for PyTorch and some distributed environments."""
    rank = 0
    world_size = 1
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        try:
            import torch.distributed

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                group = group or torch.distributed.group.WORLD
                rank = torch.distributed.get_rank(group=group)
                world_size = torch.distributed.get_world_size(group=group)
        except ModuleNotFoundError:
            pass
    return rank, world_size


def base_simple_split_by_node(urls: List[str], node_id: int = 0, node_count: int = 1):
    urls = list(urls)
    urls_split = urls[node_id::node_count]
    logger.debug(
        f"splitting {len(urls)} to {node_count} nodes. This node {node_id} gets {len(urls_split)} shards/tars."
    )
    return urls_split


def simple_split_by_node(urls: List[str]):
    rank, world_size = get_rank_world_size()
    return base_simple_split_by_node(urls, rank, world_size)

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from collections import defaultdict
from functools import partial
import json

import numpy as np

import torch

import webdataset as wds

import yaml
from atek.dataset.cubercnn_utils import to_omni3d
from atek.dataset.dataset_utils import (
    KEY_MAPPING,
    SELECTED_KEYS,
    simple_split_by_node,
    simple_split_by_worker,
)
from atek.utils.transform_utils import batch_transform_points

from torch.utils.data.dataloader import default_collate
from torchvision import transforms


# convert original 2d bbox format from xxyy to xyxy
BBOX_2D_NEW_ORDER = [0, 2, 1, 3]


def apply_map(sample, map_fn_dict=None):
    if map_fn_dict is not None:
        for key, map_fn in map_fn_dict.items():
            sample[key] = map_fn(sample[key])
    return sample


def process_sample(sample):
    """
    Process sample for image, tensor, and json data
    """
    data_dict = {}

    for k, v in sample.items():
        if k.endswith(".jpeg"):
            data_dict[k] = v

        elif k.endswith(".pth"):
            for sub_k, tensor_data in v.items():
                if isinstance(tensor_data, torch.Tensor):
                    data_dict[sub_k] = tensor_data
                elif isinstance(tensor_data, list):
                    if len(tensor_data) == 1:
                        data_dict[sub_k] = tensor_data
                    else:
                        data_dict[sub_k] = [torch.stack(tensor_data)]
                else:
                    # raise ValueError(
                    #     "Unknown type {} for key: {} and sub_key: {} at seq: {}".format(
                    #         type(tensor_data),
                    #         k,
                    #         sub_k,
                    #         sample["frame_info.json"]["F#214-1+sequence_name"],
                    #     )
                    # )
                    return None

        elif k.endswith(".json"):
            for sub_k, json_data in v.items():
                data_dict[sub_k] = json_data

    if len(SELECTED_KEYS) > 0:
        data_dict = {k: data_dict[k] for k in SELECTED_KEYS}

    if KEY_MAPPING:
        for k in KEY_MAPPING.keys():
            assert k in data_dict
        data_dict = {KEY_MAPPING.get(k, k): v for k, v in data_dict.items()}

    return data_dict


def get_eight_corners_omni3d(xmin, xmax, ymin, ymax, zmin, zmax):
    """
    Produces vertices so they will be in this order *AFTER* applying the transformation to go from
    DataEngine to Omni3D coordinates

                v4_____________________v5
                /|                    /|
               / |                   / |
              /  |                  /  |
             /___|_________________/   |
          v0|    |                 |v1 |
            |    |                 |   |
            |    |                 |   |
            |    |                 |   |
            |    |_________________|___|
            |   / v7               |   /v6
            |  /                   |  /
            | /                    | /
            |/_____________________|/
            v3                     v2
    """
    return np.array(
        [
            [xmax, ymin, zmax],
            [xmax, ymax, zmax],
            [xmax, ymax, zmin],
            [xmax, ymin, zmin],
            [xmin, ymin, zmax],
            [xmin, ymax, zmax],
            [xmin, ymax, zmin],
            [xmin, ymin, zmin],
        ]
    )


def convert_data(data_dict):
    if data_dict is None:
        return None

    data_dict_new = {}

    # RGB -> BGR
    assert len(data_dict["image"].shape) == 3
    assert data_dict["image"].shape[-1] == 3
    bgr = [2, 1, 0]
    data_dict_new["image"] = data_dict["image"][:, :, bgr]

    # add image information
    cam_param = data_dict["camera_params"][0].tolist()
    data_dict_new["K"] = [
        [cam_param[0], 0, cam_param[2]],
        [0, cam_param[1], cam_param[3]],
        [0, 0, 1],
    ]
    data_dict_new["width"] = data_dict["image"].shape[0]
    data_dict_new["height"] = data_dict["image"].shape[1]

    # add annotation information
    Ts_cam_obj = data_dict["Ts_camera_object"][0]
    data_dict_new["R_cam"] = [T_cam_obj[:, :3].tolist() for T_cam_obj in Ts_cam_obj]
    data_dict_new["center_cam"] = [T_cam_obj[:, 3].tolist() for T_cam_obj in Ts_cam_obj]

    orig_dims = data_dict["original_object_dimensions"][0]
    data_dict_new["dimensions"] = [dim.tolist()[::-1] for dim in orig_dims]

    data_dict_new["bbox2D_proj"] = data_dict["bbox_2d"][0][
        :, BBOX_2D_NEW_ORDER
    ].tolist()

    corners = [
        get_eight_corners_omni3d(
            -dim[0] / 2,
            dim[0] / 2,
            -dim[1] / 2,
            dim[1] / 2,
            -dim[2] / 2,
            dim[2] / 2,
        )
        for dim in orig_dims
    ]
    corners = torch.tensor(np.stack(corners))
    data_dict_new["bbox3D_cam"] = batch_transform_points(
        corners, Ts_cam_obj[:, :, :3], Ts_cam_obj[:, :, 3]
    ).tolist()

    # use instance ids as category ids, for instance-based detection
    data_dict_new["category_id"] = data_dict["object_instance_ids"][0]
    data_dict_new["category_name"] = [
        data_dict["category_id_to_name"][0][str(cat_id)]
        for cat_id in data_dict["object_category_ids"][0]
    ]

    # add metadata
    data_dict_new["valid3D"] = True
    data_dict_new["sequence_name"] = os.path.basename(
        os.path.dirname(data_dict["sequence_name"][0])
    )
    data_dict_new["frame_id"] = data_dict["frame_id"][0]
    data_dict_new["timestamp_ns"] = data_dict["timestamp_ns"][0]

    return data_dict_new


def data_transform(
    wds_data,
    shuffle_sample=None,
    map_fn_list=None,
    map_fn_dict=None,
):
    if shuffle_sample:
        wds_data = wds_data.shuffle(shuffle_sample)

    wds_data = wds_data.decode("rgb")

    if map_fn_list is not None:
        for map_fn in map_fn_list:
            wds_data = wds_data.map(map_fn)

    if map_fn_dict is not None:
        wds_data = wds_data.map(partial(apply_map, map_fn_dict=map_fn_dict))

    return wds_data


def adt_collate(batch):
    """
    Collate ADT data based on data types. Only collate data that are tensors
    """
    collate_keys = []
    non_collate_keys = []
    for k, v in batch[0].items():
        if isinstance(v, torch.Tensor):
            collate_keys.append(k)
        else:
            non_collate_keys.append(k)

    collated_data = {}
    non_collated_data = {}
    for key in collate_keys:
        collated_data[key] = default_collate([item[key] for item in batch])

    for key in non_collate_keys:
        non_collated_data[key] = [item[key] for item in batch]

    return {**collated_data, **non_collated_data}


def trivial_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return batch


def get_adt_wds_dataset(
    wds_tars,
    id_map,
    nodesplitter=None,
    shard_shuffle=None,
    shuffle_sample=None,
    repeat=False,
):
    if nodesplitter is not None:
        wds_data = wds.WebDataset(
            wds_tars,
            shardshuffle=shard_shuffle,
            nodesplitter=nodesplitter,
        )
    else:
        wds_data = wds.WebDataset(
            wds_tars,
            shardshuffle=shard_shuffle,
        )

    to_omni3d_partial = partial(to_omni3d, id_map=id_map)

    dataset = data_transform(
        wds_data,
        shuffle_sample,
        map_fn_list=[process_sample, convert_data, to_omni3d_partial],
        map_fn_dict=None,
    )
    if repeat:
        dataset = dataset.repeat()

    return dataset


def get_adt_dataloader(
    wds_dataset, collate_fn=trivial_collate, batch_size=4, num_workers=4
):
    dataloader = torch.utils.data.DataLoader(
        wds_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    return dataloader


def get_id_map(id_map_json):
    with open(id_map_json, "r") as f:
        id_map = json.load(f)
        id_map = {int(k): int(v) for k, v in id_map.items()}
    return id_map


def get_loader(
    tar_files,
    id_map_json,
    nodesplitter=None,
    batch_size=4,
    num_workers=4,
    shard_shuffle=None,
    repeat=False,
):
    id_map = get_id_map(id_map_json)
    adt_dataset = get_adt_wds_dataset(
        tar_files,
        id_map,
        nodesplitter=nodesplitter,
        shard_shuffle=shard_shuffle,
        repeat=repeat,
    )

    adt_dataloader = torch.utils.data.DataLoader(
        adt_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_collate,
        pin_memory=True,
    )

    return adt_dataloader

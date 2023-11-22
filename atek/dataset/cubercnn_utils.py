# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import json
import numpy as np
import torch
from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode, Instances, Keypoints


def to_omni3d(data_dict, *, id_map):
    """
    Convert data_dict to Omni3D format.
    """
    ann_keys = [
        "bbox",
        "bbox3D_cam",
        "bbox2D_proj",
        "bbox2D_trunc",
        "bbox2D_tight",
        "center_cam",
        "dimensions",
        "pose",
        "R_cam",
        "category_id",
    ]

    objs = []
    for j in range(len(data_dict["bbox2D_proj"])):
        obj = {key: data_dict[key][j] for key in ann_keys if key in data_dict}

        annotation_category_id = obj["category_id"]

        assert annotation_category_id in id_map
        annotation_category_id = id_map[annotation_category_id]

        obj["iscrowd"] = False

        # TODO: add ignore flag based on filtering settings
        ignore = False
        obj["ignore"] = ignore
        obj["ignore2D"] = ignore
        obj["ignore3D"] = ignore

        obj["bbox_mode"] = BoxMode.XYXY_ABS
        obj["bbox"] = obj["bbox2D_proj"]
        obj["pose"] = obj["R_cam"]

        obj["category_id"] = -1 if ignore else annotation_category_id
        obj["depth"] = obj["bbox3D_cam"][2]

        width, height = obj["bbox"][2] - obj["bbox"][0], obj["bbox"][3] - obj["bbox"][1]
        obj["area"] = width * height

        objs.append(obj)
    data_dict["annotations"] = objs

    return data_dict


def to_batch(data_dict):
    batch = {}
    image = (data_dict["image"] * 255.0).astype(np.uint8)
    batch["image"] = torch.tensor(image).permute(2, 0, 1)
    batch["width"] = data_dict["width"]
    batch["height"] = data_dict["height"]
    batch["K"] = data_dict["K"]

    annos = []
    for anno in data_dict["annotations"]:
        annos.append(transform_instance_anno(anno, K=data_dict["K"]))
    image_shape = batch["height"], batch["width"]
    instances = annotations_to_instances(annos, image_shape)

    batch["instances"] = detection_utils.filter_empty_instances(instances)
    batch["sequence_name"] = data_dict["sequence_name"]
    batch["frame_id"] = data_dict["frame_id"]
    batch["timestamp_ns"] = data_dict["timestamp_ns"]

    return batch


def transform_instance_anno(annotation, *, K):
    if annotation["center_cam"][2] != 0:
        point3D = annotation["center_cam"]

        point2D = K @ np.array(point3D)
        point2D[:2] = point2D[:2] / point2D[-1]
        annotation["center_cam_proj"] = point2D.tolist()

        keypoints = (K @ np.array(annotation["bbox3D_cam"]).T).T
        keypoints[:, 0] /= keypoints[:, -1]
        keypoints[:, 1] /= keypoints[:, -1]

        if annotation["ignore"]:
            # all keypoints marked as not visible
            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 1
        else:
            valid_keypoints = keypoints[:, 2] > 0

            # 0 - unknown, 1 - not visible, 2 visible
            keypoints[:, 2] = 2
            keypoints[valid_keypoints, 2] = 2

        annotation["keypoints"] = keypoints.tolist()

    return annotation


def annotations_to_instances(annos, image_size):
    # init
    target = Instances(image_size)

    # add classes, 2D boxes, 3D boxes and poses
    target.gt_classes = torch.tensor(
        [int(obj["category_id"]) for obj in annos], dtype=torch.int64
    )
    target.gt_boxes = Boxes(
        [
            BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
            for obj in annos
        ]
    )
    target.gt_boxes3D = torch.FloatTensor(
        [
            anno["center_cam_proj"] + anno["dimensions"] + anno["center_cam"]
            for anno in annos
        ]
    )
    target.gt_poses = torch.FloatTensor([anno["pose"] for anno in annos])

    # do keypoints?
    target.gt_keypoints = Keypoints(
        torch.FloatTensor([anno["keypoints"] for anno in annos])
    )

    return target

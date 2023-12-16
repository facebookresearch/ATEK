from typing import List, Optional, Dict
import torch
import os

from detectron2.data import detection_utils
from detectron2.structures import Boxes, BoxMode, Instances

from webdataset.filters import pipelinefilter
from atek.dataset.atek_webdataset import create_atek_webdataset


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
}


def omni3d_key_selection(key: str) -> bool:
    return key in KEY_MAPPING.keys()


def omni3d_key_remap(key: str) -> str:
    if key in KEY_MAPPING.keys():
        return KEY_MAPPING[key]
    else:
        return key


def get_camera_matrix(camera_params_fufvu0v0: torch.Tensor) -> torch.Tensor:
    """
    Generates the camera matrix (nx3x3) for n pinhole camera parameters n x (fx, fy, cx, cy)
    """
    assert camera_params_fufvu0v0.ndim == 2
    assert camera_params_fufvu0v0.shape[-1] == 4
    num_cameras = camera_params_fufvu0v0.shape[0]
    # Initialize an array of zeros
    camera_matrices = torch.zeros((num_cameras, 3, 3))

    # Assign fx, fy, cx, cy
    camera_matrices[:, 0, 0] = camera_params_fufvu0v0[:, 0]  # fx
    camera_matrices[:, 1, 1] = camera_params_fufvu0v0[:, 1]  # fy
    camera_matrices[:, 0, 2] = camera_params_fufvu0v0[:, 2]  # cx
    camera_matrices[:, 1, 2] = camera_params_fufvu0v0[:, 3]  # cy
    camera_matrices[:, 2, 2] = 1.

    return camera_matrices


def atek_to_omni3d(
    data,
    category_id_remapping: Optional[Dict] = None,
    min_bb2d_area=100,
    min_bb3d_depth=0.3,
    max_bb3d_depth=5,
):
    """
    A helper data transform functino to convert the ATEK webdataset data to omni3d unbatched
    samples. Yield one unbatched sample a time to use the collation and batching mechnism in 
    the webdataset properly.
    """
    for sample in data:
        # Image rgb->bgr->255->uint8
        assert sample["images"].shape[1] == 3 and sample["images"].ndim == 4
        images = (sample["images"].flip(1) * 255).to(torch.uint8)

        # Image information
        Ks = get_camera_matrix(sample["camera_params"])
        image_height, image_width = sample["images"].shape[2:]

        for idx in range(len(images)):
            sample_new = {
                "image": images[idx],
                "K": Ks[idx],
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
                    bb2ds_x0y0x1y1[:, 3] - bb2ds_x0y0x1y1[:, 1])
                filter_bb2ds_area = bb2ds_area > min_bb2d_area

                bb3ds_depth = sample["Ts_camera_object"][idx][:, -1, -1]
                filter_bb3ds_depth = (min_bb3d_depth <= bb3ds_depth) & (
                    bb3ds_depth <= max_bb3d_depth)

                # Remap the semantic id and filter classes
                # Use instance ids as category ids for instance-based detection
                sem_ids = sample["object_instance_ids"][idx]
                if category_id_remapping is not None:
                    sem_ids = [category_id_remapping.get(
                        id, -1) for id in sem_ids]
                sem_ids = torch.tensor(sem_ids)
                filter_ids = sem_ids >= 0

                final_filter = filter_bb2ds_area & filter_bb3ds_depth & filter_ids
                Ts_camera_object_filtered = sample["Ts_camera_object"][idx][final_filter]
                ts_camera_object_filtered = Ts_camera_object_filtered[:, :, 3]

                filtered_projection_2d = (sample_new["K"].repeat(len(
                    ts_camera_object_filtered), 1, 1) @ ts_camera_object_filtered.unsqueeze(-1)).squeeze(-1)
                filtered_projection_2d = filtered_projection_2d[:,
                                                                :2] / filtered_projection_2d[:, 2].unsqueeze(-1)

                instances.gt_classes = sem_ids[final_filter]
                instances.gt_boxes = Boxes(bb2ds_x0y0x1y1[final_filter])
                instances.gt_poses = Ts_camera_object_filtered[:, :3]
                instances.gt_boxes3D = torch.cat([
                    filtered_projection_2d,
                    bb3ds_depth[final_filter].unsqueeze(-1),
                    sample["object_dimensions"][idx][final_filter],
                    ts_camera_object_filtered,
                ],
                    axis=-1)

                sample_new["instances"] = instances
                yield sample_new

            # annotations = []
            # category_id_to_name = sample["category_id_to_name"][idx]
            # for T_camera_object, dimension, bb2d_x0x1y0y1, instance_id, category_id in zip(
            #     sample["Ts_camera_object"][idx],
            #     sample["object_dimensions"][idx],
            #     sample["bb2ds_x0x1y0y1"][idx],
            #     sample["object_instance_ids"][idx],
            # ):
            #     target_obj = {}
            #     target_obj["pose"] = T_camera_object[:, :3] # R_camera_object
            #     target_obj["center_cam"] = T_camera_object[:, 3] # t_camera_object
            #     target_obj["dimensions"]= dimension

            #     # Remap the semantic id and filter classes
            #     sem_id = instance_id # Use instance ids as category ids for instance-based detection
            #     if category_id_remapping is not None:
            #         if sem_id not in category_id_remapping.keys():
            #             continue
            #         else:
            #            sem_id = category_id_remapping[sem_id]
            #     target_obj["category_id"] = sem_id
            #     target_obj["category_name"] = category_id_to_name[category_id]

            #     # Change the bb2d format to x0y0x1y1
            #     target_obj["bbox_mode"]= BoxMode.XYXY_ABS
            #     target_obj["bbox"].append(bb2d_x0x1y0y1.numpy()[0, 2, 1, 3])

            #     # Filter obj with small bb2d area
            #     bb2d_area = (target_obj["bbox"][2] - target_obj["bbox"][0]) * (target_obj["bbox"][3] - target_obj["bbox"][1])
            #     if bb2d_area < min_bbox2d_area:
            #         continue

            #     # Filter obj with depth range
            #     depth = target_obj["center_cam"][2]
            #     if depth < min_bbox3d_depth or depth > max_bbox3d_depth:
            #         continue

            #     # project the 3D box annotation XYZ_3D to screen
            #     t_camera_object = target_obj['center_cam']
            #     object_center_in_camera = K @ np.array(t_camera_object)
            #     object_center_in_camera[:2] = object_center_in_camera[:2] / object_center_in_camera[-1]
            #     target_obj["center_cam_proj"] = object_center_in_camera[:2]

            #     annotations.append(target_obj)

            # sample_new["instances"] = annotations_to_instances(annos, sample_new["image"].shape[2:])

            # yield sample_new


def collate_as_list(batch):
    return list(batch)


def create_omni3d_webdataset(
    urls: List,
    batch_size: int,
    category_id_remapping: Optional[Dict] = None,
    min_bb2d_area=100,
    min_bb3d_depth=0.3,
    max_bb3d_depth=5,
):
    return create_atek_webdataset(
        urls,
        batch_size,
        select_key_fn=omni3d_key_selection,
        remap_key_fn=omni3d_key_remap,
        data_transform_fn=pipelinefilter(atek_to_omni3d)(
            category_id_remapping, min_bb2d_area, min_bb3d_depth, max_bb3d_depth),
        collation_fn=collate_as_list,
    )

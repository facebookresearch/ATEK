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

from typing import Dict, List, Optional

import numpy as np

import torch

import webdataset as wds
from atek.util.atek_constants import ATEK_CATEGORY_ID_TO_NAME, ATEK_CATEGORY_NAME_TO_ID
from detectron2.structures import Boxes, Instances
from projectaria_tools.core.sophus import SE3

from surreal.data_services.atek.atek.data_loaders.atek_wds_dataloader import (
    load_atek_wds_dataset,
)

from webdataset.filters import pipelinefilter


class CubeRCNNModelAdaptor:
    def __init__(
        self,
        # TODO: make these a DictConfig
        min_bb2d_area: Optional[float] = 100,
        min_bb3d_depth: Optional[float] = 0.3,
        max_bb3d_depth: Optional[float] = 5.0,
    ):
        self.min_bb2d_area = min_bb2d_area
        self.min_bb3d_depth = min_bb3d_depth
        self.max_bb3d_depth = max_bb3d_depth

    @staticmethod
    def get_dict_key_mapping_all():
        dict_key_mapping = {
            "mfcd#camera-rgb+images": "image",
            "mfcd#camera-rgb+projection_params": "camera_params",
            "mfcd#camera-rgb+camera_model_name": "camera_model",
            "mfcd#camera-rgb+t_device_camera": "t_device_rgbcam",
            "mfcd#camera-rgb+frame_ids": "frame_id",
            "mfcd#camera-rgb+capture_timestamps_ns": "timestamp_ns",
            "mtd#ts_world_device": "ts_world_device",
            "sequence_name": "sequence_name",
            "gt_data": "gt_data",
        }
        return dict_key_mapping

    def atek_to_cubercnn(self, data):
        """
        A helper data transform function to convert a ATEK webdataset data sample built by CubeRCNNSampleBuilder, to CubeRCNN unbatched
        samples. Yield one unbatched sample a time to use the collation and batching mechanism in
        the webdataset properly.
        """
        for atek_wds_sample in data:
            sample = {}
            self._update_camera_data_in_sample(atek_wds_sample, sample)
            self._update_T_world_camera(atek_wds_sample, sample)

            # Skip if no gt data
            if "gt_data" in atek_wds_sample and len(atek_wds_sample["gt_data"]) > 0:
                self._update_gt_data_in_sample(atek_wds_sample, sample)

            yield sample

    def _update_camera_data_in_sample(self, atek_wds_sample, sample):
        """
        Initialize sample image
        Process camera K-matrix information and update the sample dictionary.
        """
        assert atek_wds_sample["image"].shape[0] == 1, "Only support 1 frame"
        image_height, image_width = atek_wds_sample["image"].shape[2:]

        # calculate K-matrix
        camera_model = atek_wds_sample["camera_model"]
        assert (
            camera_model == "CameraModelType.LINEAR"
        ), f"Only linear camera model supported in CubeRCNN model, this data has {camera_model} instead."
        k_matrix = torch.zeros((3, 3), dtype=torch.float32)
        params = atek_wds_sample["camera_params"]
        k_matrix[0, 0], k_matrix[1, 1] = params[0], params[1]
        k_matrix[0, 2], k_matrix[1, 2] = params[2], params[3]
        k_matrix[2, 2] = 1.0

        sample.update(
            {
                # rgb -> bgr
                "image": atek_wds_sample["image"][0, [2, 1, 0], :, :].clone().detach(),
                "K": k_matrix.tolist(),
                "height": image_height,
                "width": image_width,
                "K_matrix": k_matrix,
                "timestamp_ns": atek_wds_sample["timestamp_ns"],
                "frame_id": atek_wds_sample["frame_id"],
                "sequence_name": atek_wds_sample["sequence_name"],
            }
        )

    def _update_T_world_camera(self, atek_wds_sample, sample):
        """
        Compute world-to-camera transformation matrices, and update this field in sample dict.
        """
        T_world_device = SE3.from_matrix3x4(atek_wds_sample["ts_world_device"][0])
        T_device_rgbCam = SE3.from_matrix3x4(atek_wds_sample["t_device_rgbcam"])
        T_world_rgbCam = T_world_device @ T_device_rgbCam
        sample["T_world_camera"] = T_world_rgbCam.to_matrix3x4()

    def _process_2d_bbox_dict(self, bb2d_dict):
        """
        Process 2D bounding boxes by rearranging the bounding box coordinates to be
        in the order x0, y0, x1, y1 and calculating the area of each 2D bounding box.
        """
        bb2ds_x0y0x1y1 = bb2d_dict["box_ranges"]
        bb2ds_x0y0x1y1 = bb2ds_x0y0x1y1[:, [0, 2, 1, 3]]
        bb2ds_area = (bb2ds_x0y0x1y1[:, 2] - bb2ds_x0y0x1y1[:, 0]) * (
            bb2ds_x0y0x1y1[:, 3] - bb2ds_x0y0x1y1[:, 1]
        )

        return bb2ds_x0y0x1y1, bb2ds_area

    def _process_3d_bbox_dict(self, bbox3d_dict, T_world_rgbCam):
        """
        This function processes 3D bounding box data from a given dictionary,
        extracting dimensions, calculating depths, and computing transformation
        matrices relative to the camera.
        """
        bb3d_dimensions = bbox3d_dict["object_dimensions"]

        bb3d_depths_list = []
        Ts_world_object_list = []
        Ts_cam_object_list = []
        for _, pose_as_tensor in enumerate(bbox3d_dict["ts_world_object"]):
            T_world_object = SE3.from_matrix3x4(pose_as_tensor.numpy())
            T_cam_object = T_world_rgbCam.inverse() @ T_world_object

            # Add to lists
            Ts_world_object_list.append(
                torch.tensor(T_world_object.to_matrix3x4(), dtype=torch.float32)
            )

            Ts_cam_object_list.append(
                torch.tensor(T_cam_object.to_matrix3x4(), dtype=torch.float32)
            )
            bb3d_depths_list.append(T_cam_object.translation()[:, 2].item())

        # Convert lists to tensors
        bb3d_depths = torch.tensor(bb3d_depths_list, dtype=torch.float32)
        Ts_world_object = torch.stack(Ts_world_object_list, dim=0)
        Ts_cam_object = torch.stack(Ts_cam_object_list, dim=0)

        return bb3d_dimensions, bb3d_depths, Ts_world_object, Ts_cam_object

    def _update_gt_data_in_sample(self, atek_wds_sample, sample):
        """
        updates the sample dictionary with filtered ground truth data for both 2D and 3D bounding boxes.
        """
        bbox2d_dict = atek_wds_sample["gt_data"]["obb2_gt"]["camera-rgb"]
        bbox3d_dict = atek_wds_sample["gt_data"]["obb3_gt"]["camera-rgb"]

        # Instance id between obb3 and obb2 should be the same
        assert torch.allclose(
            bbox3d_dict["instance_ids"], bbox2d_dict["instance_ids"], atol=0
        ), "instance ids in obb2 and obb3 needs to be exactly the same!"

        category_ids = bbox3d_dict["category_ids"]

        T_world_rgbCam = SE3.from_matrix3x4(sample["T_world_camera"])

        bb2ds_x0y0x1y1, bb2ds_area = self._process_2d_bbox_dict(bbox2d_dict)
        bb3d_dimensions, bb3d_depths, Ts_world_object, Ts_cam_object = (
            self._process_3d_bbox_dict(bbox3d_dict, T_world_rgbCam)
        )

        # Filter 1: ignore category = -1, meaning "Other".
        category_id_filter = category_ids > 0  # filter out -1 category = "Other"

        # Filter 2: ignore bboxes with small area
        bb2d_area_filter = bb2ds_area > self.min_bb2d_area

        # Filter 3: ignore bboxes with small depth
        bb3d_depth_filter = (self.min_bb3d_depth <= bb3d_depths) & (
            bb3d_depths <= self.max_bb3d_depth
        )

        # Combine all filters
        final_filter = category_id_filter & bb2d_area_filter & bb3d_depth_filter

        # Apply filter to create instances
        image_height = sample["height"]
        image_width = sample["width"]
        instances = Instances((image_height, image_width))
        instances.gt_classes = category_ids[final_filter]
        instances.gt_boxes = Boxes(bb2ds_x0y0x1y1[final_filter])

        # Create 3D bboxes
        Ts_cam_object_filtered = Ts_cam_object[final_filter]
        trans_cam_object_filtered = Ts_cam_object_filtered[:, :, 3]
        k_matrix = sample["K_matrix"]
        filtered_projection_2d = (
            k_matrix.repeat(len(trans_cam_object_filtered), 1, 1)
            @ trans_cam_object_filtered.unsqueeze(-1)
        ).squeeze(-1)
        filtered_projection_2d = filtered_projection_2d[:, :2] / filtered_projection_2d[
            :, 2
        ].unsqueeze(-1)
        instances.gt_boxes3D = torch.cat(
            [
                filtered_projection_2d,  # [N, 2]
                bb3d_depths[final_filter].unsqueeze(-1).clone().detach(),  # [N, 1]
                # Omni3d has the inverted zyx dimensions
                # https://github.com/facebookresearch/omni3d/blob/main/cubercnn/util/math_util.py#L144C1-L181C40
                bb3d_dimensions[final_filter].flip(-1).clone().detach(),  # [N, 3]
                trans_cam_object_filtered,  # [N, 3]
            ],
            axis=-1,
        )
        instances.gt_poses = Ts_cam_object_filtered[:, :, :3].clone().detach()

        # Update sample with filtered instance data
        sample["instances"] = instances
        sample["Ts_world_object"] = Ts_world_object[final_filter].clone().detach()
        sample["object_dimensions"] = bb3d_dimensions[final_filter].clone().detach()
        sample["category"] = category_ids[final_filter].clone().detach()

    @staticmethod
    def cubercnn_gt_to_atek_gt(
        cubercnn_dict: Dict,
        T_world_camera_np: np.array,
        camera_label: str = "camera-rgb",
        cubercnn_id_to_atek_id: Optional[Dict[int, int]] = None,
    ) -> Optional[Dict]:
        """
        A helper data transform function to convert the model input (gt) dict, or output (prediction) dict from CubeRCNN format,
        back to ATEK GT dict format (defined in `obb_sample_builder`, which is effectively obb3_gt_processor + obb2_gt_processor)
        CubeRCNN model is ran only on one camera stream, so user should specific which camera stream to use. By default, it is "camera-rgb".
        """
        cubercnn_instances = cubercnn_dict["instances"]
        # Skip if no instances
        if len(cubercnn_instances) == 0:
            return None

        # Check if the cubercnn_dict is a prediction dict or gt dict. If it is a gt dict,
        # "transfer" it to a prediction dict by filling in the pred fields. TODO: Consider another way to handle this!
        pred_flag = hasattr(cubercnn_instances, "pred_classes")
        if not pred_flag:
            # fake pred fields using gt fields
            num_instances = len(cubercnn_instances.gt_classes)
            cubercnn_instances.pred_classes = cubercnn_instances.gt_classes
            cubercnn_instances.pred_boxes = cubercnn_instances.gt_boxes
            cubercnn_instances.pred_dimensions = cubercnn_instances.gt_boxes3D[:, 3:6]
            cubercnn_instances.pred_center_cam = cubercnn_instances.gt_boxes3D[:, 6:9]
            cubercnn_instances.pred_pose = cubercnn_instances.gt_poses
            cubercnn_instances.scores = torch.ones(num_instances, dtype=torch.float32)

        # initialize ATEK GT dict
        atek_dict = {
            "obb3_gt": {},
            "obb2_gt": {},
            "scores": cubercnn_instances.scores.detach().cpu(),  # tensor, shape: [num_instances], float32
        }
        atek_dict["obb3_gt"][camera_label] = {
            "instance_ids": None,
            "category_names": None,
            "category_ids": cubercnn_instances.pred_classes.detach().cpu(),
        }
        atek_dict["obb2_gt"][camera_label] = {
            "instance_ids": None,
            "category_names": None,
            "category_ids": cubercnn_instances.pred_classes.detach().cpu(),
            "visibility_ratios": None,
        }

        # Fill in category ids
        if cubercnn_id_to_atek_id is not None:
            atek_id_list = [
                cubercnn_id_to_atek_id[id.item()]
                for id in cubercnn_instances.pred_classes
            ]
            atek_dict["obb3_gt"][camera_label]["category_ids"] = torch.tensor(
                atek_id_list, dtype=torch.int32
            )
            atek_dict["obb2_gt"][camera_label]["category_ids"] = torch.tensor(
                atek_id_list, dtype=torch.int32
            )
        else:
            atek_dict["obb3_gt"][camera_label][
                "category_ids"
            ] = cubercnn_instances.pred_classes.detach().cpu()
            atek_dict["obb2_gt"][camera_label][
                "category_ids"
            ] = cubercnn_instances.pred_classes.detach().cpu()

        # Fill category names
        atek_dict["obb3_gt"][camera_label]["category_names"] = [
            ATEK_CATEGORY_ID_TO_NAME[id.item()]
            for id in atek_dict["obb3_gt"][camera_label]["category_ids"]
        ]
        atek_dict["obb2_gt"][camera_label]["category_names"] = [
            ATEK_CATEGORY_ID_TO_NAME[id.item()]
            for id in atek_dict["obb2_gt"][camera_label]["category_ids"]
        ]

        # CubeRCNN dimensions are in reversed order (zyx) compared to ATEK (xyz)
        bbox3d_dim = (
            cubercnn_instances.pred_dimensions.detach().cpu()
        )  # tensor, shape [num_instances, 3]
        atek_dict["obb3_gt"][camera_label]["object_dimensions"] = torch.flip(
            bbox3d_dim, dims=[1]
        )

        # Fill in pose
        rotations = cubercnn_instances.pred_pose.detach().cpu()  # [num_instances, 3, 3]
        translations = (
            cubercnn_instances.pred_center_cam.detach().cpu().unsqueeze(2)
        )  # [num_instances, 3, 1]

        Ts_cam_object = SE3.from_matrix3x4(
            torch.cat((rotations, translations), dim=2).numpy()
        )
        T_world_cam = SE3.from_matrix3x4(T_world_camera_np)

        Ts_world_object = T_world_cam @ Ts_cam_object
        Ts_world_object = SE3.to_matrix3x4(Ts_world_object)  # [num_instances, 3, 4]
        if Ts_world_object.shape == (3, 4):
            Ts_world_object = Ts_world_object.reshape(1, 3, 4)
        atek_dict["obb3_gt"][camera_label]["ts_world_object"] = torch.tensor(
            Ts_world_object, dtype=torch.float32
        )

        # Fill in 2d bbox ranges
        bbox2d = (
            cubercnn_instances.pred_boxes.tensor.detach().cpu()
        )  # tensor, shape [num_instances, 4]
        # x0,y0,x1,y1 -> x0,x1,y0,y1
        atek_dict["obb2_gt"][camera_label]["box_ranges"] = torch.stack(
            (bbox2d[:, 0], bbox2d[:, 2], bbox2d[:, 1], bbox2d[:, 3]), dim=1
        )

        return atek_dict


def cubercnn_collation_fn(batch):
    # Simply collate as a list
    return list(batch)


def load_atek_wds_dataset_as_cubercnn(
    urls: List, batch_size: Optional[int], repeat_flag: bool, shuffle_flag: bool = False
) -> wds.WebDataset:
    cubercnn_model_adaptor = CubeRCNNModelAdaptor()

    return load_atek_wds_dataset(
        urls,
        batch_size=batch_size,
        dict_key_mapping=CubeRCNNModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(cubercnn_model_adaptor.atek_to_cubercnn)(),
        collation_fn=cubercnn_collation_fn,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )


def create_atek_dataloader_as_cubercnn(
    urls: List[str],
    batch_size: Optional[int] = None,
    repeat_flag: bool = False,
    shuffle_flag: bool = False,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    wds_dataset = load_atek_wds_dataset_as_cubercnn(
        urls,
        batch_size=batch_size,
        repeat_flag=repeat_flag,
        shuffle_flag=shuffle_flag,
    )

    return torch.utils.data.DataLoader(
        wds_dataset, batch_size=None, num_workers=num_workers, pin_memory=True
    )

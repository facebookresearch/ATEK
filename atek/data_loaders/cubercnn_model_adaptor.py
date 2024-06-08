# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List, Optional

import torch

from atek.data_loaders.atek_wds_dataloader import load_atek_wds_dataset
from detectron2.structures import Boxes, Instances
from projectaria_tools.core.sophus import SE3

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
            "gtdata": "gtdata",
        }
        return dict_key_mapping

    def atek_to_cubercnn(self, data):
        """
        A helper data transform function to convert a ATEK webdataset data sample built by CubeRCNNSampleBuilder, to CubeRCNN unbatched
        samples. Yield one unbatched sample a time to use the collation and batching mechanism in
        the webdataset properly.
        """
        for atek_wds_sample in data:
            sample = atek_wds_sample
            # for testing only
            print(f"atek_wds_sample keys: {sample.keys()}")

            # check frame number should be 1
            assert (
                sample["image"].shape[0] == 1
            ), "cubercnn sample currently only support 1 frame"
            # Image rgb->bgr
            sample["image"] = sample["image"][:, [2, 1, 0], :, :]

            # Formulating camera K-matrix information from projection params
            camera_model = sample["camera_model"]
            assert (
                camera_model == "CameraModelType.LINEAR"
            ), f"CubeRCNN model only take camera model of Linear, this data has {camera_model} instead."
            k_matrix = torch.zeros((3, 3), dtype=torch.float32)
            k_matrix[0, 0] = sample["camera_params"][0]  # fx
            k_matrix[1, 1] = sample["camera_params"][1]  # fy
            k_matrix[0, 2] = sample["camera_params"][2]  # cx
            k_matrix[1, 2] = sample["camera_params"][3]  # cy
            k_matrix[2, 2] = 1.0
            # images are [1, C, H, W]
            image_height, image_width = sample["image"].shape[2:]
            sample.update(
                {
                    "K": k_matrix.tolist(),  # CubeRCNN requires list input
                    "height": image_height,
                    "width": image_width,
                }
            )

            # Compute T_world_camera
            T_world_device = SE3.from_matrix3x4(sample["ts_world_device"][0])
            T_device_rgbCam = SE3.from_matrix3x4(sample["t_device_rgbcam"])
            T_world_rgbCam = T_world_device @ T_device_rgbCam
            sample["T_world_camera"] = T_world_rgbCam.to_matrix3x4()

            # retrieve gt data from the 2 dicts
            bbox2d_dict = sample["gtdata"]["obb2_gt"]["camera-rgb"]
            bbox3d_dict = sample["gtdata"]["obb3_gt"]
            shared_instances = set(bbox2d_dict.keys()) & set(bbox3d_dict.keys())
            bb3d_dimensions = torch.tensor(
                [bbox3d_dict[inst]["object_dimensions"] for inst in shared_instances],
                dtype=torch.float32,
            )
            category_ids = torch.tensor(
                [int(bbox3d_dict[inst]["category_id"]) for inst in shared_instances],
                dtype=torch.int64,
            )
            box_range_list = [
                bbox2d_dict[inst]["box_range"] for inst in shared_instances
            ]

            # Filter 1: ignore category = -1, meaning "Other".
            category_id_filter = category_ids > 0  # filter out -1 category = "Other"

            # Filter 2: ignore bboxes with small area
            # [xmin, xmax, ymin, ymax] -> [xmin, ymin, xmax, ymax]
            bb2ds_x0y0x1y1 = torch.tensor(box_range_list, dtype=torch.float32)
            bb2ds_x0y0x1y1 = bb2ds_x0y0x1y1[:, [0, 2, 1, 3]]
            bb2ds_area = (bb2ds_x0y0x1y1[:, 2] - bb2ds_x0y0x1y1[:, 0]) * (
                bb2ds_x0y0x1y1[:, 3] - bb2ds_x0y0x1y1[:, 1]
            )
            bb2d_area_filter = bb2ds_area > self.min_bb2d_area

            # Filter 3: ignore bboxes with small depth
            bb3d_depths_list = []
            Ts_world_object_list = []
            Ts_cam_object_list = []
            for inst in shared_instances:
                single_bb3d_dict = bbox3d_dict[inst]
                T_world_object = SE3.from_matrix3x4(single_bb3d_dict["T_World_Object"])
                T_cam_object = T_world_rgbCam.inverse() @ T_world_object

                # Add to lists
                Ts_world_object_list.append(
                    torch.tensor(T_world_object.to_matrix3x4(), dtype=torch.float32)
                )
                Ts_cam_object_list.append(
                    torch.tensor(T_cam_object.to_matrix3x4(), dtype=torch.float32)
                )
                bb3d_depths_list.append(
                    torch.tensor(T_cam_object.translation()[:, 2], dtype=torch.float32)
                )
            # Convert lists to tensors
            bb3d_depths = torch.stack(bb3d_depths_list, dim=0).squeeze()
            Ts_world_object = torch.stack(Ts_world_object_list, dim=0)
            Ts_cam_object = torch.stack(Ts_cam_object_list, dim=0)

            bb3d_depth_filter = (self.min_bb3d_depth <= bb3d_depths) & (
                bb3d_depths <= self.max_bb3d_depth
            )

            # Combine all filters
            final_filter = category_id_filter & bb2d_area_filter & bb3d_depth_filter

            # Apply filter to create instances
            instances = Instances((image_height, image_width))
            instances.gt_classes = category_ids[final_filter]
            instances.gt_boxes = Boxes(bb2ds_x0y0x1y1[final_filter])

            # Create 3D bboxes
            Ts_cam_object_filtered = Ts_cam_object[final_filter]
            trans_cam_object_filtered = Ts_cam_object_filtered[:, :, 3]
            filtered_projection_2d = (
                k_matrix.repeat(len(trans_cam_object_filtered), 1, 1)
                @ trans_cam_object_filtered.unsqueeze(-1)
            ).squeeze(-1)
            filtered_projection_2d = filtered_projection_2d[
                :, :2
            ] / filtered_projection_2d[:, 2].unsqueeze(-1)
            instances.gt_boxes3D = torch.cat(
                [
                    filtered_projection_2d,
                    bb3d_depths[final_filter].unsqueeze(-1).clone().detach(),
                    # Omni3d has the inverted zyx dimensions
                    # https://github.com/facebookresearch/omni3d/blob/main/cubercnn/util/math_util.py#L144C1-L181C40
                    bb3d_dimensions[final_filter].flip(-1).clone().detach(),
                    trans_cam_object_filtered,
                ],
                axis=-1,
            )
            instances.gt_poses = Ts_cam_object_filtered[:, :, :3].clone().detach()

            sample["instances"] = instances

            # Other fields in sample
            sample["Ts_world_object"] = Ts_world_object[final_filter].clone().detach()
            sample["object_dimensions"] = bb3d_dimensions[final_filter].clone().detach()
            sample["category"] = category_ids[final_filter].clone().detach()

            yield sample


def load_atek_wds_dataset_as_cubercnn(urls: List):
    cubercnn_model_adaptor = CubeRCNNModelAdaptor()

    return load_atek_wds_dataset(
        urls,
        dict_key_mapping=CubeRCNNModelAdaptor.get_dict_key_mapping_all(),
        data_transform_fn=pipelinefilter(cubercnn_model_adaptor.atek_to_cubercnn)(),
    )

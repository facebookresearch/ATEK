import numpy as np
import torch
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider

from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor


class AtekCubercnnDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, selected_device_number=0):
        paths_provider = AriaDigitalTwinDataPathsProvider(data_path)
        data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)
        rotate_image_cw90deg = True

        pose_data_processor = PoseDataProcessor(
            name="pose",
            trajectory_file=data_paths.aria_trajectory_filepath,
        )
        rgb_data_processor = FrameDataProcessor(
            video_vrs=data_paths.aria_vrs_filepath,
            stream_id=StreamId("214-1"),
            rotate_image_cw90deg=rotate_image_cw90deg,
            target_linear_camera_params=np.array([512, 512]),
            pose_data_processor=pose_data_processor,
            gt_data_processor=None,
        )

        self.data_processor = rgb_data_processor

    def __len__(self):
        return len(self.data_processor)

    def __getitem__(self, index):
        rgb_image_frame = self.data_processor.get_frame_by_index(index)

        cam_param = rgb_image_frame.camera_parameters
        K = np.array(
            [
                [cam_param[0], 0, cam_param[2]],
                [0, cam_param[1], cam_param[3]],
                [0, 0, 1],
            ]
        )
        image = rgb_image_frame.image
        if format == "BGR":
            image = image[:, :, [2, 1, 0]]

        batched = [
            {
                "data_source": rgb_image_frame.data_source,
                "sequence_name": rgb_image_frame.sequence_name,
                "index": index,
                "frame_id": rgb_image_frame.frame_id,
                "timestamp_ns": rgb_image_frame.timestamp_ns,
                "T_world_cam": rgb_image_frame.T_world_camera,
                "image": torch.as_tensor(
                    np.ascontiguousarray(image.transpose(2, 0, 1))
                ),
                "height": image.shape[0],
                "width": image.shape[1],
                "K": K,
            }
        ]

        return batched


def build_cubercnn_dataset(data_path, selected_device_number=0):
    return AtekCubercnnDataset(data_path, selected_device_number=selected_device_number)

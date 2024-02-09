from typing import Dict, List
import numpy as np
import torch

from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider

from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor
from atek.dataset.common import AriaStreamIds


def linear_camera_params_to_kMat(cam_param: np.ndarray) -> np.ndarray:
    """
    Convert linear camera parameters to a 3x3 matrix:
    """
    kMat = np.array(
        [
            [cam_param[0], 0, cam_param[2]],
            [0, cam_param[1], cam_param[3]],
            [0, 0, 1],
        ]
    )
    return kMat


class AtekCubercnnInferDataset(torch.utils.data.Dataset):
    """
    Dataset class for ATEK CubeRCNN model inference pipeline (for ADT/ASE data).

    This dataset class processes the original ADT data format (VRS) into format consumable
    by CubeRCNN model (https://github.com/facebookresearch/omni3d). Specifically, by default,
    the video sequence is processed into a sequence of image frames, each containing the RGB
    image with the camera pose and additional metadata.

    Args:
        data_path (str): root directory for a sequence.
        selected_device_number (int): device number to use, there could be multiple device's
            (e.g. N) recordings in the same dataset (sequence). This number indicate which device's
            recording is chosen to process (0 to N-1).
        stream_id (AriaStreamIds): stream id to use, rgb_stream_id, slaml_stream_id, or
            slamr_stream_id.
        rotate_image_cw90deg (bool): whether to rotate image clockwise by 90 degrees.
        target_image_resolution (List[int]): image resolution (width, height) after processing.
        target_color_format (str): image format after processing, "RGB" or "BGR".
    """

    def __init__(
        self,
        data_path: str,
        selected_device_number: int = 0,
        stream_id: AriaStreamIds = AriaStreamIds.rgb_stream_id,
        rotate_image_cw90deg: bool = True,
        target_image_resolution: List[int] = [512, 512],
        target_color_format: str = "BGR",
    ):
        paths_provider = AriaDigitalTwinDataPathsProvider(data_path)
        data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)

        pose_data_processor = PoseDataProcessor(
            name="pose",
            trajectory_file=data_paths.aria_trajectory_filepath,
        )
        rgb_data_processor = FrameDataProcessor(
            video_vrs=data_paths.aria_vrs_filepath,
            stream_id=stream_id,
            rotate_image_cw90deg=rotate_image_cw90deg,
            target_linear_camera_params=np.array(
                [target_image_resolution[0], target_image_resolution[1]]
            ),
            pose_data_processor=pose_data_processor,
            gt_data_processor=None,
        )

        self.data_processor = rgb_data_processor
        self.target_color_format = target_color_format

    def __len__(self) -> int:
        return len(self.data_processor)

    def __getitem__(self, index: int) -> List[Dict]:
        rgb_image_frame = self.data_processor.get_frame_by_index(index)

        cam_param = rgb_image_frame.camera_parameters
        K = linear_camera_params_to_kMat(cam_param)
        image = rgb_image_frame.image
        if self.target_color_format == "BGR":
            image = image[:, :, [2, 1, 0]]

        # image dimension (height, width, channel) to (channel, height, width)
        image = image.transpose(2, 0, 1)

        batched = [
            {
                "data_source": rgb_image_frame.data_source,
                "sequence_name": rgb_image_frame.sequence_name,
                "index": index,
                "frame_id": rgb_image_frame.frame_id,
                "timestamp_ns": rgb_image_frame.timestamp_ns,
                "T_world_cam": rgb_image_frame.T_world_camera,
                "image": torch.as_tensor(np.ascontiguousarray(image)),
                "height": image.shape[1],
                "width": image.shape[2],
                "K": K,
            }
        ]

        return batched

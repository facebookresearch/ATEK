from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import AriaDigitalTwinDataPathsProvider

from atek.data_preprocess.adt_gt_data_processor import AdtGtDataProcessor
from atek.data_preprocess.frame_data_processor import FrameDataProcessor
from atek.data_preprocess.pose_data_processor import PoseDataProcessor


class AriaStreamIds:
    rgb_stream_id = StreamId("214-1")
    slaml_stream_id = StreamId("1201-1")
    slamr_stream_id = StreamId("1201-2")


class AtekRawFrameDataset(torch.utils.data.Dataset):
    """
    Dataset class to access raw ATEK data (ADT/ASE).

    This dataset class uses the data processor to convert the original ADT/ASE data format (VRS)
    into sequence of Frame's (as in data_preprocess/data_schema.py). Each Frame contains the RGB,
    SLAM-L, SLAM-R images with object annotations and other metadata. Additionally, this dataset
    can take some Callable to convert Frame into another format, e.g., dict, for downstream use.

    Args:
        data_path (str): root directory for a sequence.
        selected_device_number (int): device number to use, there could be multiple device's
            (e.g. N) recordings in the same dataset (sequence). This number indicate which device's
            recording is chosen to process (0 to N-1).
        stream_id (AriaStreamIds): stream id to use, rgb_stream_id, slaml_stream_id, or
            slamr_stream_id.
        rotate_image_cw90deg (bool): whether to rotate image clockwise by 90 degrees.
        target_image_resolution (List[int]): image resolution (width, height) after processing.
        transform_fn (Callable): a Callable to convert Frame into another format
    """

    def __init__(
        self,
        data_path: str,
        selected_device_number: int = 0,
        stream_id: AriaStreamIds = AriaStreamIds.rgb_stream_id,
        rotate_image_cw90deg: bool = True,
        target_image_resolution: Tuple[int] = (512, 512),
        transform_fn: Optional[Callable] = None,
    ):
        paths_provider = AriaDigitalTwinDataPathsProvider(data_path)
        data_paths = paths_provider.get_datapaths_by_device_num(selected_device_number)

        pose_data_processor = PoseDataProcessor(
            name="pose",
            trajectory_file=data_paths.aria_trajectory_filepath,
        )
        try:
            rgb_adt_gt_data_processor = AdtGtDataProcessor(
                "adt_gt", "ADT", stream_id, data_paths
            )
        except:
            rgb_adt_gt_data_processor = None
            print(f"Ground-truth NOT available for sequence: {data_path}")
        rgb_data_processor = FrameDataProcessor(
            video_vrs=data_paths.aria_vrs_filepath,
            stream_id=stream_id,
            rotate_image_cw90deg=rotate_image_cw90deg,
            target_linear_camera_params=np.array(
                [target_image_resolution[0], target_image_resolution[1]]
            ),
            pose_data_processor=pose_data_processor,
            gt_data_processor=rgb_adt_gt_data_processor,
        )

        self.data_processor = rgb_data_processor
        self.transform_fn = transform_fn

    def __len__(self) -> int:
        return len(self.data_processor)

    def __getitem__(self, index: int) -> List[Dict]:
        rgb_image_frame = self.data_processor.get_frame_by_index(index)

        if self.transform_fn is None:
            return rgb_image_frame
        else:
            return self.transform_fn(rgb_image_frame)

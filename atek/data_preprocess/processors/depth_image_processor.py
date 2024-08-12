# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

import torch

from atek.data_preprocess.atek_data_sample import MultiFrameCameraData

from omegaconf.omegaconf import DictConfig
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from projectaria_tools.core.stream_id import StreamId
from torchvision.transforms import InterpolationMode, v2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DepthImageProcessor:
    def __init__(
        self,
        depth_vrs: str,
        image_transform: Callable,
        conf: DictConfig,
    ):
        # Parse in conf
        self.conf = conf
        self.image_transform = image_transform

        # setting up vrs data provider
        self.depth_vrs = depth_vrs
        self.stream_id = StreamId(conf.sensor_stream_id)
        self.data_provider = self.setup_vrs_data_provider()

        # Cache capture timestamps
        self.time_domain = getattr(TimeDomain, conf.time_domain)
        self.camera_timestamps = self.data_provider.get_timestamps_ns(
            self.stream_id, self.time_domain
        )

    def setup_vrs_data_provider(self):
        """
        Setup the vrs data provider, and only activate the stream id specified by camera_label.
        Returns: vrs_data_provider, stream_id
        """
        provider = data_provider.create_vrs_data_provider(self.depth_vrs)
        assert provider is not None, f"Cannot open {self.depth_vrs}"
        assert (
            self.stream_id in provider.get_all_streams()
        ), f"Specified stream id {self.stream_id} is not in vrs {self.depth_vrs}'s stream id list: {provider.get_all_streams()}"

        return provider

    def get_depth_data_by_timestamps_ns(
        self, timestamps_ns: List[int]
    ) -> Optional[MultiFrameCameraData]:
        """
        Obtain images by timestamps. Image should be processed.
        returns: if successful, returns (image_data: Tensor [numFrames, numChannel, height, width], capture_timestamp: Tensor[numFrames], frame_id_in_stream: Tensor[numFrames])
                else returns None
        """
        image_list = []
        capture_timestamp_list = []
        frame_id_list = []
        for single_timestamp in timestamps_ns:
            index = self.data_provider.get_index_by_time_ns(
                stream_id=self.stream_id,
                time_ns=single_timestamp,
                time_domain=self.time_domain,
                time_query_options=TimeQueryOptions.CLOSEST,
            )
            image_data_and_record = self.data_provider.get_image_data_by_index(
                self.stream_id, index
            )
            frame_id = image_data_and_record[1].frame_number
            capture_timestamp = image_data_and_record[1].capture_timestamp_ns

            # Check if fetched frame is within tolerance
            if abs(capture_timestamp - single_timestamp) > self.conf.tolerance_ns:
                continue

            # Handle uint16 not supported by torch
            np_image = image_data_and_record[0].to_numpy_array()
            if np_image.dtype == np.uint16:
                np_image = np_image.astype(np.uint32)
            image = torch.from_numpy(np_image)
            if len(image.shape) == 2:
                # single channel image: [h,w] -> [c, h, w]
                image = torch.unsqueeze(image, dim=0)
            else:
                raise ValueError("Depth image is expected to have single channel")

            image_list.append(image)
            capture_timestamp_list.append(capture_timestamp)
            frame_id_list.append(frame_id)
        # End for single_timestamp

        # Check if at least one frame is successfully fetched
        if len(image_list) == 0:
            return None

        # Image transformations are handled by torchvision's transform functions.
        batched_depth_tensor = torch.stack(image_list, dim=0)
        batched_depth_tensor = self.image_transform(batched_depth_tensor)

        # properly clean output to desired dtype and shapes
        result = MultiFrameCameraData(
            images=batched_depth_tensor,
            capture_timestamps_ns=torch.tensor(
                capture_timestamp_list, dtype=torch.int64
            ),
            frame_ids=torch.tensor(frame_id_list, dtype=torch.int64),
        )
        return result

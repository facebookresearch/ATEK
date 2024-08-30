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
        camera_label: str,
        conf: DictConfig,
    ):
        # Parse in conf
        self.conf = conf
        self.image_transform = image_transform
        self.camera_label = camera_label

        # setting up vrs data provider
        self.depth_vrs = depth_vrs
        self.data_provider, self.stream_id = self.setup_vrs_data_provider()

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
        assert (
            provider is not None
        ), f"Cannot open depth vrs under path [{self.depth_vrs}]"

        # Find depth stream in provider
        result_stream_id = None
        # If conf specified `depth_stream_id`, use it directly
        if "depth_stream_id" in self.conf:
            result_stream_id = StreamId(self.conf.depth_stream_id)
            assert (
                result_stream_id in provider.get_all_streams()
            ), f"Specified stream id {result_stream_id} is not in vrs {self.depth_vrs}'s stream id list: {provider.get_all_streams()}"
        # If conf specified `depth_stream_type_id`, check if any of the existing streams match the type id, as $TYPE_ID-*
        elif "depth_stream_type_id" in self.conf:
            for stream_id in provider.get_all_streams():
                stream_numeric_name = str(stream_id)
                type_numeric_name = stream_numeric_name.split("-")[
                    0
                ]  # obtain the "214" out of "214-1"
                if type_numeric_name == self.conf.depth_stream_type_id:
                    result_stream_id = stream_id

                    break
            # check valid stream id is found
            assert (
                result_stream_id is not None
            ), f"Specified stream type id {self.conf.depth_stream_type_id} is not in vrs {self.depth_vrs}'s stream id list: {provider.get_all_streams()}"
        # If none of the above
        else:
            raise ValueError("No depth stream id or type id is specified in config")

        return provider, result_stream_id

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
                np_image = np_image.astype(np.float32)

            # Convert depth units from mm to meters, if specified in config
            if "unit_scaling" in self.conf:
                np_image = np_image * self.conf.unit_scaling

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

        # Fill in camera calibration information
        result.camera_label = self.camera_label
        return result

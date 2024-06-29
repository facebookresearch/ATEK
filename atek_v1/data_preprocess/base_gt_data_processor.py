# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from abc import ABC, abstractmethod
from typing import Optional

from atek_v1.data_preprocess.data_schema import Frame

from projectaria_tools.core import calibration

from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseGtDataProcessor(ABC):
    def __init__(
        self,
        name: str,
        stream_id: StreamId,
    ):
        """
        name: data type readable name
        """
        self.name = name
        self.data_source = "Unknown"
        self.stream_id = stream_id

    @abstractmethod
    def set_undistortion_params(
        self,
        original_camera_calibration: calibration.CameraCalibration,
        target_camera_calibration: Optional[calibration.CameraCalibration] = None,
        rotate_image_cw90deg: bool = True,
    ):
        pass

    @abstractmethod
    def get_object_gt_at_timestamp_ns(
        self,
        frame: Frame,
        timestamp_ns: int,
        T_camera_world: SE3,
        stream_id: Optional[StreamId] = None,
        tolerance_ns: int = 1000_000,
    ):
        pass

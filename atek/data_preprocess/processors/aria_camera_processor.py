# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
from typing import List, Optional, Tuple

import torch

from omegaconf.omegaconf import DictConfig
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from torchvision import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AriaCameraProcessor:
    def __init__(
        self,
        video_vrs: str,
        conf: DictConfig,
    ):
        # Parse in conf
        self.conf = conf
        self.target_camera_resolution: List = (
            conf.target_camera_resolution if "target_camera_resolution" in conf else []
        )
        self.undistort_to_linear_camera: bool = (
            conf.undistort_to_linear_camera
            if "undistort_to_linear_camera" in conf
            else False
        )
        self.rotate_image_cw90deg: bool = (
            conf.rotate_image_cw90deg if "rotate_image_cw90deg" in conf else False
        )

        # setting up vrs data provider
        self.video_vrs = video_vrs
        self.camera_label = conf.sensor_label
        self.data_provider, self.stream_id = self.setup_vrs_data_provider()

        # setting up camera calibration, and optional linear camera (for rectification)
        self.camera_calibration = self.setup_camera_calibration()

        # Cache capture timestamps
        self.time_domain = getattr(TimeDomain, conf.time_domain)
        self.camera_timestamps = self.data_provider.get_timestamps_ns(
            self.stream_id, self.time_domain
        )

    def get_final_camera_calib(self):
        return self.final_camera_calib

    def get_origin_label(self):
        """
        returns the sensor label of the origin (DeviceFrame) definition in Aria calibration
        """
        return self.data_provider.get_device_calibration().get_origin_label()

    class DistortByCalibrationTVWrapper:
        """
        A torch vision transform function wrapper on top of `calirbation.distort_by_calibration()` function,
        so that this operation can be chained with other tv.transforms.
        """

        def __init__(self, dstCalib, srcCalib):
            self.dstCalib = dstCalib
            self.srcCalib = srcCalib

        def __call__(self, image):
            return calibration.distort_by_calibration(
                image, self.dstCalib, self.srcCalib
            )

    def setup_vrs_data_provider(self):
        """
        Setup the vrs data provider, and only activate the stream id specified by camera_label.
        Returns: vrs_data_provider, stream_id
        """
        provider = data_provider.create_vrs_data_provider(self.video_vrs)
        assert provider is not None, f"Cannot open {self.video_vrs}"
        stream_id = provider.get_stream_id_from_label(self.camera_label)
        assert stream_id is not None, f"Cannot find stream id for {self.camera_label}"

        return provider, stream_id

    def setup_camera_calibration(
        self,
    ):
        """
        Setup camera calibration from the VRS, which is done through the following steps (IN ORDER!):
        1. obtain camera calibration from vrs, this corresponds to the VRS resolution.
        2. (if `undistort_to_linear_camera` is True) undistort the camera calibration to a linear camera calibration,
                where focal is kept the same as `focal_x` value in the original intrinsics.
        3. (if `target_camera_resolution` or `target_linear_camera_focal` are specified) convert camera calibration to a linear camera calibration
        4. (if `rotate_image_cw90deg` is True) rotate the camera calibration (both intrinsics and extrinsics) by 90deg clockwise.
            Note that the `target_camera_resolution` and `target_linear_camera_params` should be set according to the UNROTATED camera.
        """
        # obtain the camera calibration from vrs
        stream_sensor_calib = self.data_provider.get_sensor_calibration(self.stream_id)
        assert (
            stream_sensor_calib is not None
        ), f"Can not get the sensor calibration for {self.stream_id} in vrs {self.video_vrs}"
        camera_calib = stream_sensor_calib.camera_calibration()
        self.original_camera_calib = camera_calib

        # undistort to linear camera if specified
        if self.undistort_to_linear_camera:
            camera_calib = calibration.get_linear_camera_calibration(
                image_width=camera_calib.get_image_size()[0],
                image_height=camera_calib.get_image_size()[1],
                focal_length=camera_calib.get_focal_lengths()[0],
                label=f"undistorted_linear_{camera_calib.get_label()}",
                T_Device_Camera=camera_calib.get_transform_device_camera(),
            )
            self.undistorted_linear_camera_calib = (
                camera_calib  # save this for image undistortion operations
            )

        # rescale resolution if specified
        if (
            self.target_camera_resolution is not None
            and len(self.target_camera_resolution) == 2
        ):
            scale = (
                float(self.target_camera_resolution[0])
                / camera_calib.get_image_size()[0]
            )
            camera_calib = camera_calib.rescale(
                new_resolution=self.target_camera_resolution, scale=scale
            )

        # Rotate camera model if specified
        if self.rotate_image_cw90deg:
            camera_calib = calibration.rotate_camera_calib_cw90deg(camera_calib)

        self.final_camera_calib = camera_calib

    def get_image_data_by_timestamp_ns(
        self, timestamp_ns: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Obtain a single frame image by timestamp. Image should be processed.
        returns: if successful, returns (image_data: Tensor [numChannel, height, width], capture_timestamp: Tensor[1,], frame_id_in_stream: Tensor[1,])
                else returns None
        """
        index = self.data_provider.get_index_by_time_ns(
            stream_id=self.stream_id,
            time_ns=timestamp_ns,
            time_domain=self.time_domain,
            time_query_options=TimeQueryOptions.CLOSEST,
        )
        image_data_and_record = self.data_provider.get_image_data_by_index(
            self.stream_id, index
        )
        frame_id = image_data_and_record[1].frame_number
        capture_timestamp = image_data_and_record[1].capture_timestamp_ns

        # Check if fetched frame is within tolerance
        if abs(capture_timestamp - timestamp_ns) > self.conf.tolerance_ns:
            return None

        image = torch.from_numpy(image_data_and_record[0].to_numpy_array())
        if len(image.shape) == 2:
            # single channel image: [h,w] -> [c, h, w]
            image = torch.unsqueeze(image, dim=0)
        else:
            # rgb image: [h, w, c] -> [c, h ,w]
            image = image.permute(2, 0, 1)

        # Image transformations are handled by torchvision's transform functions.
        image_transform_list = []
        # undistort if specified
        if self.undistort_to_linear_camera:
            image_transform_list.append(
                self.DistortByCalibrationTVWrapper(
                    dstCalib=self.undistorted_linear_camera_calib,
                    srcCalib=self.original_camera_calib,
                )
            )

        if (
            self.target_camera_resolution is not None
            and len(self.target_camera_resolution) == 2
        ):
            image_transform_list.append(
                transforms.Resize(self.target_camera_resolution)
            )

        if self.rotate_image_cw90deg:
            # image is [c, h, w]
            image_transform_list.append(lambda img: torch.rot90(img, k=3, dims=[1, 2]))

        image_transform = transforms.Compose(image_transform_list)

        transformed_image = image_transform(image)

        # properly clean output to desired dtype and shapes
        return (
            torch.unsqueeze(transformed_image, dim=0),
            torch.tensor([capture_timestamp], dtype=torch.int64),
            torch.tensor([frame_id], dtype=torch.int64),
        )

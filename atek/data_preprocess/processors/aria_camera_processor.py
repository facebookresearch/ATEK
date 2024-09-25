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

import logging
from typing import Callable, List, Optional, Tuple

import torch

from atek.data_preprocess.atek_data_sample import MultiFrameCameraData
from atek.util.camera_calib_utils import (
    rescale_pixel_coords,
    rotate_pixel_coords_cw90,
    undistort_pixel_coords,
)

from omegaconf.omegaconf import DictConfig
from PIL import Image
from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions  # @manual
from torchvision.transforms import InterpolationMode, v2

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def identity_transform(obj):
    """Helper identity transform function"""
    return obj


class AriaCameraProcessor:
    def __init__(
        self,
        video_vrs: str,
        conf: DictConfig,  # TODO: consider use more explicit init, and overload with DictConfig
    ):
        # Parse in conf
        self.conf = conf
        # Resolution-rescale related params
        self.target_camera_resolution: List = (
            conf.target_camera_resolution if "target_camera_resolution" in conf else []
        )
        self.rescale_antialias: bool = (
            conf.rescale_antialias if "rescale_antialias" in conf else True
        )
        # Camera model undistortion related params
        self.undistort_to_linear_camera: bool = (
            conf.undistort_to_linear_camera
            if "undistort_to_linear_camera" in conf
            else False
        )
        # Image rotation related params
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

    def get_stream_id(self):
        return self.stream_id

    def get_origin_label(self):
        """
        returns the sensor label of the origin (DeviceFrame) definition in Aria calibration
        """
        return self.data_provider.get_device_calibration().get_origin_label()

    def setup_vrs_data_provider(self):
        """
        Setup the vrs data provider, and only activate the stream id specified by camera_label.
        Returns: vrs_data_provider, stream_id
        """
        provider = data_provider.create_vrs_data_provider(self.video_vrs)
        assert (
            provider is not None
        ), f"Cannot open video.vrs file under path [{self.video_vrs}]"
        stream_id = provider.get_stream_id_from_label(self.camera_label)
        assert (
            stream_id is not None
        ), f"Cannot find stream id for camera [{self.camera_label}]"

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
            self.scale = (
                float(self.target_camera_resolution[0])
                / camera_calib.get_image_size()[0]
            )
            camera_calib = camera_calib.rescale(
                new_resolution=self.target_camera_resolution, scale=self.scale
            )

        # Rotate camera model if specified
        if self.rotate_image_cw90deg:
            camera_calib = calibration.rotate_camera_calib_cw90deg(camera_calib)

        self.final_camera_calib = camera_calib

    class DistortByCalibrationTVWrapper:
        """
        A torch vision transform function wrapper on top of `calibration.distort_by_calibration()` function,
        so that this operation can be chained with other tv.transforms.
        """

        def __init__(self, dstCalib, srcCalib, is_transforming_label_data=False):
            self.dstCalib = dstCalib
            self.srcCalib = srcCalib
            self.is_transforming_label_data = is_transforming_label_data

        def __call__(self, image):
            if image.ndim != 4:
                raise ValueError(
                    f"Expecting 4D tensor of [Frame, C, H, W], got {image.ndim}D tensor instead."
                )
            if image.dtype in [torch.int32]:
                # projectaria_tools calibration will cast int32 or uint32 to uint8 but leaves uint64 as uint64
                image = image.to(torch.uint64)

            num_frames = image.size(0)

            result_list = []
            for i in range(num_frames):
                # input image is tensor shape of [Frame, C, H, W], while distort_by_calibration requires [H, W, C]
                single_image = image[i].permute(1, 2, 0).numpy().copy()

                if self.is_transforming_label_data:
                    numpy_result = calibration.distort_label_by_calibration(
                        single_image, self.dstCalib, self.srcCalib
                    )
                else:
                    numpy_result = calibration.distort_by_calibration(
                        single_image, self.dstCalib, self.srcCalib
                    )
                tensor_result = torch.from_numpy(numpy_result)
                if tensor_result.ndim == 2:
                    # [H, W] -> [1, H, W]
                    tensor_result = tensor_result.unsqueeze(0)
                else:
                    # [H, W, C] -> [C, H, W]
                    tensor_result = tensor_result.permute(2, 0, 1)

                result_list.append(tensor_result)

            return torch.stack(result_list, dim=0)

    def get_image_transform(
        self,
        rescale_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        is_transforming_label_data: bool = False,
    ) -> Callable:
        """
        Returns a list of torchvision transform functions to be applied to the raw image data,
        in the order of: undistortion -> rescale -> rotateCW90, where any step is optional.
        If transforming label data (as is the case with segmentation masks), the interpolation
        should be set to nearest, and is_transforming_label_data should be set to True.
        """
        image_transform_list = []
        # undistort if specified
        if self.undistort_to_linear_camera:
            image_transform_list.append(
                self.DistortByCalibrationTVWrapper(
                    dstCalib=self.undistorted_linear_camera_calib,
                    srcCalib=self.original_camera_calib,
                    is_transforming_label_data=is_transforming_label_data,
                )
            )

        if (
            self.target_camera_resolution is not None
            and len(self.target_camera_resolution) == 2
        ):
            # resolution is specified as [w, h]. Need to pass [h,w] here
            image_transform_list.append(
                v2.Resize(
                    [
                        self.target_camera_resolution[1],
                        self.target_camera_resolution[0],
                    ],
                    interpolation=rescale_interpolation,
                    antialias=self.rescale_antialias,
                )
            )

        if self.rotate_image_cw90deg:
            # image is [frames, c, h, w]
            image_transform_list.append(lambda img: torch.rot90(img, k=3, dims=[2, 3]))

        if len(image_transform_list) == 0:
            return identity_transform
        else:
            return v2.Compose(image_transform_list)

    def get_pixel_transform(self) -> Callable:
        """
        Returns a list of torchvision transform functions to be applied to the raw image data,
        in the order of: undistortion -> rescale -> rotateCW90, where any step is optional.
        The transform should be applied to pixel coordinates of Tensor [N, 2]
        """
        pixel_transform_list = []
        # undistort if specified
        if self.undistort_to_linear_camera:
            pixel_transform_list.append(
                lambda pixels: undistort_pixel_coords(
                    pixels,
                    src_calib=self.original_camera_calib,
                    dst_calib=self.undistorted_linear_camera_calib,
                )
            )

        # rescale resolution if specified
        if (
            self.target_camera_resolution is not None
            and len(self.target_camera_resolution) == 2
        ):

            pixel_transform_list.append(
                lambda pixels: rescale_pixel_coords(pixels, scale=self.scale)
            )

        # rotate if specified
        if self.rotate_image_cw90deg:
            # image is [frames, c, h, w]
            pixel_transform_list.append(
                lambda pixels: rotate_pixel_coords_cw90(
                    pixels, image_dim_after_rot=self.final_camera_calib.get_image_size()
                )
            )

        if len(pixel_transform_list) == 0:
            return identity_transform
        else:
            return v2.Compose(pixel_transform_list)

    def get_image_data_by_timestamps_ns(
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
        exposure_list = []
        gain_list = []
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

            # reshape image to proper tensor shape
            image = torch.from_numpy(image_data_and_record[0].to_numpy_array())

            if len(image.shape) == 2:
                # single channel image: [h,w] -> [c, h, w]
                image = torch.unsqueeze(image, dim=0)
            else:
                # rgb image: [h, w, c] -> [c, h ,w]
                image = image.permute(2, 0, 1)
            image_list.append(image)

            # insert other values from the image data
            image_record = image_data_and_record[1]
            exposure_list.append(image_record.exposure_duration)
            gain_list.append(image_record.gain)
            capture_timestamp_list.append(capture_timestamp)
            frame_id_list.append(frame_id)
        # End for single_timestamp

        # Check if at least one frame is successfully fetched
        if len(image_list) == 0:
            return None

        # Image transformations are handled by torchvision's transform functions.
        batched_image_tensor = torch.stack(image_list, dim=0)

        image_transform = self.get_image_transform()
        batched_image_tensor = image_transform(batched_image_tensor)

        # properly clean output to desired dtype and shapes
        result = MultiFrameCameraData(
            images=batched_image_tensor,
            capture_timestamps_ns=torch.tensor(
                capture_timestamp_list, dtype=torch.int64
            ),
            frame_ids=torch.tensor(frame_id_list, dtype=torch.int64),
            exposure_durations_s=torch.tensor(exposure_list, dtype=torch.float32),
            gains=torch.tensor(gain_list, dtype=torch.float32),
        )

        return result

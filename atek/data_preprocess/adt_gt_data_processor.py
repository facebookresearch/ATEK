# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging

from functools import partial

from typing import Optional

import numpy as np
from atek.data_preprocess.base_gt_data_processor import BaseGtDataProcessor
from atek.data_preprocess.data_schema import Frame
from atek.data_preprocess.data_utils import insert_and_check
from projectaria_tools.core import calibration

from projectaria_tools.core.sophus import SE3
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPaths,
    AriaDigitalTwinDataProvider,
    bbox2d_to_image_coordinates,
)
from toolz import compose

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def identity_transform(obj):
    """Helper identity transform function"""
    return obj


def sample_edge(p1: np.ndarray, p2: np.ndarray, n):
    """Helper function to sample edge points between two points in N equidistant samples"""
    return np.column_stack((np.linspace(p1[0], p2[0], n), np.linspace(p1[1], p2[1], n)))


def center_object_bb3d(bb3d_aabb: np.ndarray, T_world_bb3d: np.ndarray):
    """
    Helper function to transform the object coordinate to the object center
    and generate the new T_world_object with new object coordinate
    bb3d_aabb: [xmin, xmax, ymin, ymax, zmin, zmax]
    """
    object_dimension = bb3d_aabb[1::2] - bb3d_aabb[::2]
    t_center_in_object = (bb3d_aabb[1::2] + bb3d_aabb[::2]) / 2

    T_bb3d_object_center = SE3.from_quat_and_translation(
        1, np.array([0, 0, 0]), t_center_in_object
    )
    T_world_object_center = T_world_bb3d @ T_bb3d_object_center

    return object_dimension, T_world_object_center


def undistort_bb2d(
    bb2d: np.ndarray,
    src_calib: calibration.CameraCalibration,
    dst_calib: calibration.CameraCalibration,
    num_points_on_edge: int = 10,
):
    """bb2d is a np array of [x_min x_max y_min y_max] format"""
    bb2d_image_coordinates = bbox2d_to_image_coordinates(bb2d)
    points_on_edges = np.row_stack(
        [
            sample_edge(
                bb2d_image_coordinates[i],
                bb2d_image_coordinates[(i + 1) % 4],
                num_points_on_edge,
            )
            for i in range(4)
        ]
    )

    rect_bb2d_xs = []
    rect_bb2d_ys = []
    for points in points_on_edges:
        unprojected_bbox2d_ray = src_calib.unproject_no_checks(points)
        rect_bb2d_coord = dst_calib.project_no_checks(unprojected_bbox2d_ray)
        rect_bb2d_xs.append(rect_bb2d_coord[0])
        rect_bb2d_ys.append(rect_bb2d_coord[1])

    dst_image_width, dst_image_height = dst_calib.get_image_size()
    rect_bb2d_min_x = min(max(min(rect_bb2d_xs), 0.0), dst_image_width - 1.0)
    rect_bb2d_max_x = min(max(max(rect_bb2d_xs), 0.0), dst_image_width - 1.0)
    rect_bb2d_min_y = min(max(min(rect_bb2d_ys), 0.0), dst_image_height - 1.0)
    rect_bb2d_max_y = min(max(max(rect_bb2d_ys), 0.0), dst_image_height - 1.0)

    rect_bb2d = np.array(
        [
            rect_bb2d_min_x,
            rect_bb2d_max_x,
            rect_bb2d_min_y,
            rect_bb2d_max_y,
        ]
    )

    return rect_bb2d


def rotate_bb2d_cw90(
    bb2d: np.ndarray,
    image_height: int,
):
    new_x_min = image_height - 1.0 - bb2d[3]
    new_x_max = image_height - 1.0 - bb2d[2]
    new_y_min = bb2d[0]
    new_y_max = bb2d[1]
    return np.array([new_x_min, new_x_max, new_y_min, new_y_max])


class AdtGtDataProcessor(BaseGtDataProcessor):
    def __init__(
        self,
        name: str,
        data_source: str,
        stream_id: StreamId,
        data_path: AriaDigitalTwinDataPaths,
    ):
        super().__init__(name, stream_id)
        self.data_srouce = data_source
        self.gt_provider = AriaDigitalTwinDataProvider(data_path)
        self.bb2d_transform_fn = identity_transform

    def set_undistortion_params(
        self,
        original_camera_calibration: calibration.CameraCalibration,
        target_camera_calibration: Optional[calibration.CameraCalibration] = None,
        rotate_image_cw90deg: bool = True,
    ):
        """
        Set the camera calibration transformation setting to the gt generator.
        The same transformation should be used to undistort the images. This will
        be applied to the 2d bounding box gt processing. Please make sure this is
        properly set before calling `get_object_gt_at_timestamp_ns`.
        """
        image_width, image_height = original_camera_calibration.get_image_size()
        if target_camera_calibration is None:
            bb2d_undistort_fn = identity_transform
        else:
            image_width, image_height = target_camera_calibration.get_image_size()
            bb2d_undistort_fn = partial(
                undistort_bb2d,
                src_calib=original_camera_calibration,
                dst_calib=target_camera_calibration,
                num_points_on_edge=10,
            )

        if rotate_image_cw90deg:
            bb2d_rotate_fn = partial(rotate_bb2d_cw90, image_height=image_height)
        else:
            bb2d_rotate_fn = identity_transform

        self.bb2d_transform_fn = compose(bb2d_rotate_fn, bb2d_undistort_fn)

    def get_object_gt_at_timestamp_ns(
        self,
        frame: Frame,
        timestamp_ns: int,
        T_camera_world: SE3,
        tolerance_ns: int = 1000_0000,
    ):
        # Getting 2d bounding boxes from GT first. We must need to have 2d info first to
        # find the 3d bounding box visibility. Default by querying the closest timestamp.
        bbox2d_with_dt = self.gt_provider.get_object_2d_boundingboxes_by_timestamp_ns(
            timestamp_ns,
            self.stream_id,
        )
        if not bbox2d_with_dt.is_valid():
            logger.warn(
                f"Can not get good 2d bounding boxes at {timestamp_ns} because input time is invalid"
            )
            return

        if bbox2d_with_dt.dt_ns() > tolerance_ns:
            logger.warn(
                f"Can not get good 2d bounding boxes at {timestamp_ns} because "
                f"the nearest bb2d with delta time {bbox2d_with_dt.dt_ns()}ns "
                f"bigger than the threshold we have {tolerance_ns}ns "
            )
            return

        # Get 3d bounding boxes data
        bbox3d_with_dt = self.gt_provider.get_object_3d_boundingboxes_by_timestamp_ns(
            timestamp_ns,
        )
        bb3d_available = True
        if not bbox3d_with_dt.is_valid():
            logger.warn(
                f"Can not get good 3d bounding boxes at {timestamp_ns} because input time is invalid"
            )
            bb3d_available = False

        if bbox3d_with_dt.dt_ns() > tolerance_ns:
            logger.warn(
                f"Can not get good 3d bounding boxes at {timestamp_ns} because "
                f"the nearest bb2d with delta time {bbox3d_with_dt.dt_ns()}ns "
                f"bigger than the threshold we have {tolerance_ns}ns "
            )
            bb3d_available = False

        bbox2d_all = bbox2d_with_dt.data()

        frame.category_id_to_name = {}

        frame.object_instance_ids = []
        frame.object_category_ids = []
        frame.bb2ds = []

        frame.object_dimensions = []
        frame.Ts_world_object = []
        frame.Ts_camera_object = []

        for instance_id, bbox2d_data in bbox2d_all.items():
            instance_info = self.gt_provider.get_instance_info_by_id(instance_id)
            category_name = instance_info.category
            category_id = instance_info.category_uid

            insert_and_check(frame.category_id_to_name, category_id, category_name)
            frame.object_instance_ids.append(instance_id)
            frame.object_category_ids.append(category_id)
            frame.bb2ds.append(self.bb2d_transform_fn(bbox2d_data.box_range))

            if bb3d_available:
                bbox3d_data = bbox3d_with_dt.data()[instance_id]
                bb3d_aabb = bbox3d_data.aabb
                T_world_bb3d = bbox3d_data.transform_scene_object
                object_dimension, T_world_object = center_object_bb3d(
                    bb3d_aabb, T_world_bb3d
                )

                frame.object_dimensions.append(object_dimension)
                frame.Ts_world_object.append(T_world_object.to_matrix3x4())
                frame.Ts_camera_object.append(
                    (T_camera_world @ T_world_object).to_matrix3x4()
                )

        assert (
            len(frame.object_instance_ids)
            == len(frame.object_category_ids)
            == len(frame.bb2ds)
        )
        if bb3d_available:
            assert (
                len(frame.object_category_ids)
                == len(frame.object_dimensions)
                == len(frame.Ts_world_object)
                == len(frame.Ts_camera_object)
            ), (
                f"{len(frame.object_category_ids)}, {len(frame.object_dimensions)}, "
                f"{len(frame.Ts_world_object)}, {len(frame.Ts_camera_object)}"
            )

    def get_bb3d_at_timestamps_ns(self):
        return self.rate_hz

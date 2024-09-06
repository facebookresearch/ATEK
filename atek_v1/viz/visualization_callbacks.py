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

from typing import Dict, List

import numpy as np
import rerun as rr
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun import ToTransform3D


TRAJECTORY_COLOR = [30, 100, 30]
GT_COLOR = [30, 200, 30]
PRED_COLOR = [200, 30, 30]


class AtekInferViewer:
    """
    Viewer for the ATEK model inference pipeline, which visualizes model predictions for each
    frame, including RGB image, 3D and 2D bounding boxes. GT 3D and 2D bounding boxes are also
    visualized if available.

    Args:
        web_port (int): port number for the web server
        ws_port (int): port number for the websocket server
        camera_name (str): name of the camera to be visualized
    """

    def __init__(self, web_port: int, ws_port: int, camera_name: str):
        self.camera_name = camera_name
        rr.init("ATEK CubeRCNN Inference Viewer", spawn=True)
        rr.serve(web_port=web_port, ws_port=ws_port)

    def _log_camera_and_image(self, input):
        # image transform: shape CHW -> HWC, color BGR-> RGB
        image = input["image"].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]

        # log camera
        camera_param = input["K"]
        rr.log(
            f"world/device/{self.camera_name}",
            rr.Pinhole(
                resolution=[
                    image.shape[1],  # width
                    image.shape[0],  # height
                ],
                focal_length=float(camera_param[0][0]),
            ),
            timeless=True,
        )

        # log camera pose
        T_world_cam = SE3.from_matrix3x4(input["T_world_camera"])
        rr.log(
            f"world/device/{self.camera_name}",
            ToTransform3D(T_world_cam, False),
        )

        # log image
        rr.log(
            f"world/device/{self.camera_name}/image",
            rr.Image(image),
        )

    def _log_gt_3d_2d_bbox(self, input):
        # log GT 3D bounding boxes
        bb3ds_sizes_gt = input["object_dimensions"]
        Ts_world_object = [SE3.from_matrix3x4(T) for T in input["Ts_world_object"]]
        bb3ds_centers_gt = [T.translation()[0] for T in Ts_world_object]
        wxyz = [T.rotation().to_quat()[0] for T in Ts_world_object]
        bb3ds_quats_xyzw_gt = [[q[3], q[0], q[1], q[2]] for q in wxyz]
        bb2ds_XYXY = [
            [bb2d[0], bb2d[2], bb2d[1], bb2d[3]] for bb2d in input["bb2ds_x0x1y0y1"]
        ]
        labels_gt = input["category"]
        rr.log(
            f"world/device/bb3d_gt/{self.camera_name}",
            rr.Boxes3D(
                sizes=bb3ds_sizes_gt,
                centers=bb3ds_centers_gt,
                rotations=bb3ds_quats_xyzw_gt,
                radii=0.01,
                colors=GT_COLOR,
                labels=labels_gt,
            ),
        )

        # log GT 2D bounding boxes
        rr.log(
            f"world/device/{self.camera_name}/bb2d_gt",
            rr.Boxes2D(
                array=bb2ds_XYXY,
                array_format=rr.Box2DFormat.XYXY,
                radii=1,
                colors=GT_COLOR,
                labels=labels_gt,
            ),
        )

    def _log_pred_3d_2d_bbox(self, input, prediction):
        T_world_cam = SE3.from_matrix3x4(input["T_world_camera"])
        bb2ds_XYXY_infer = []
        labels_infer = []
        bb3ds_centers_infer = []
        bb3ds_quats_xyzw_infer = []
        bb3ds_sizes_infer = []

        for pred in prediction:
            T_cam_obj_mat = np.zeros([3, 4])
            T_cam_obj_mat[0:3, 0:3] = np.array(pred["R_cam_obj"])
            T_cam_obj_mat[:, 3] = pred["t_cam_obj"]
            T_cam_obj = SE3.from_matrix3x4(T_cam_obj_mat)
            T_world_obj = T_world_cam @ T_cam_obj

            bb2ds_XYXY_infer.append(pred["bbox_2D"])
            labels_infer.append(pred["category"])
            bb3ds_centers_infer.append(T_world_obj.translation()[0])
            wxyz = T_world_obj.rotation().to_quat()[0]
            bb3ds_quats_xyzw_infer.append([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])
            bb3ds_sizes_infer.append(np.array(pred["dimensions"]))

        # log 3D bounding boxes
        rr.log(
            f"world/device/bb3d_infer/{self.camera_name}",
            rr.Boxes3D(
                sizes=bb3ds_sizes_infer,
                centers=bb3ds_centers_infer,
                rotations=bb3ds_quats_xyzw_infer,
                radii=0.01,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

        # log 2D bounding boxes
        rr.log(
            f"world/device/{self.camera_name}/bb2d_infer",
            rr.Boxes2D(
                array=bb2ds_XYXY_infer,
                array_format=rr.Box2DFormat.XYXY,
                radii=1,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

    def __call__(self, model_input: List[Dict], model_prediction: List[List[Dict]]):
        assert len(model_input) == len(model_prediction)

        for input, prediction in zip(model_input, model_prediction):
            rr.set_time_nanos("frame_time_ns", input["timestamp_ns"])

            # log pinhole camera and image
            self._log_camera_and_image(input)

            # log ground truth 3D and 2D bounding boxes
            if "Ts_world_object" in input:
                self._log_gt_3d_2d_bbox(input)

            # log predicted 3D and 2D bounding boxes
            if len(prediction) > 0:
                self._log_pred_3d_2d_bbox(input, prediction)

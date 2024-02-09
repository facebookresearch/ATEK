from argparse import Namespace
from typing import Dict, List

import numpy as np
import rerun as rr
from detectron2.config import CfgNode
from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun import ToTransform3D
from atek.dataset.cubercnn_data import AtekCubercnnInferDataset


from atek.viz.viz import AtekViewer

TRAJECTORY_COLOR = [30, 100, 30]
GT_COLOR = [30, 200, 30]
PRED_COLOR = [200, 30, 30]


class AtekCubercnnInferViewer(AtekViewer):
    """
    Viewer for ATEK CubeRCNN model inference pipeline, which visualizes model predictions for each
    frame, including RGB image, 3D and 2D bounding boxes. Camera trajectory and pose are also
    visualized.

    Args:
        dataset (AtekCubercnnInferDataset): ATEK CubeRCNN inference dataset
        config (Dict): config to create the CubeRCNN inference viewer
    """

    def __init__(self, dataset: AtekCubercnnInferDataset, config: Dict):
        self.dataset = dataset
        self.data_processor = dataset.data_processor
        self.camera_name = self.data_processor.vrs_camera_calib.get_label()

        rr.init("ATEK CubeRCNN Inference Viewer", spawn=True)
        rr.serve(web_port=config["web_port"], ws_port=config["ws_port"])

        # log trajectory
        trajectory = [
            self.data_processor.get_T_world_camera_by_index(i).translation()[0]
            for i in range(len(self.data_processor))
        ]
        rr.log(
            f"world/device/trajectory",
            rr.LineStrips3D(trajectory, colors=TRAJECTORY_COLOR, radii=0.01),
            timeless=True,
        )

        # log pinhole camera
        camera_calib = dataset.data_processor.final_camera_calib
        rr.log(
            f"world/device/{self.camera_name}",
            rr.Pinhole(
                resolution=[
                    int(camera_calib.get_image_size()[0]),
                    int(camera_calib.get_image_size()[1]),
                ],
                focal_length=float(camera_calib.get_focal_lengths()[0]),
            ),
            timeless=True,
        )

    def __call__(
        self, data: List[Dict], prediction: List, args: Namespace, cfg: CfgNode
    ):
        assert len(data) == 1
        ts = data[0]["timestamp_ns"]
        rr.set_time_nanos("frame_time_ns", ts)

        # process 3D and 2D bounding boxes
        T_world_cam = SE3.from_matrix3x4(data[0]["T_world_cam"])
        bb2ds_XYXY_infer = []
        labels_infer = []
        bb3ds_centers_infer = []
        bb3ds_sizes_infer = []

        if len(prediction) == 0:
            print("No prediction!")
            return

        for pred in prediction:
            T_cam_obj_mat = np.zeros([3, 4])
            T_cam_obj_mat[0:3, 0:3] = np.array(pred["R_cam_obj"])
            T_cam_obj_mat[:, 3] = pred["t_cam_obj"]
            T_cam_obj = SE3.from_matrix3x4(T_cam_obj_mat)
            T_world_obj = T_world_cam @ T_cam_obj

            bb2ds_XYXY_infer.append(pred["bbox_2D"])
            labels_infer.append(pred["category"])
            bb3ds_centers_infer.append(T_world_obj.translation()[0])
            bb3ds_sizes_infer.append(np.array(pred["dimensions"]))

        # log camera pose
        rr.log(
            f"world/device/{self.camera_name}",
            ToTransform3D(T_world_cam, False),
        )

        # log 3D bounding boxes
        rr.log(
            f"world/device/bb3d_infer",
            rr.Boxes3D(
                sizes=bb3ds_sizes_infer,
                centers=bb3ds_centers_infer,
                radii=0.01,
                colors=PRED_COLOR,
                labels=labels_infer,
            ),
        )

        # log image
        image = data[0]["image"].detach().cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        rr.log(
            f"world/device/{self.camera_name}/image",
            rr.Image(image),
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

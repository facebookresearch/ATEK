# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import logging
from typing import Dict, List, Optional

import rerun as rr
import torch

from atek.data_loaders.cubercnn_model_adaptor import CubeRCNNModelAdaptor
from atek.util.viz_utils import filter_obbs_by_confidence
from atek.viz.atek_visualizer_base import NativeAtekSampleVisualizer
from omegaconf.omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CubercnnVisualizer(NativeAtekSampleVisualizer):
    """
    A visualizer class to plot CubeRCNN-format dict data
    """

    def __init__(
        self,
        viz_prefix: str = "",
        viz_web_port: Optional[int] = None,
        conf: Optional[DictConfig] = None,
        output_viz_file: Optional[str] = None,
    ) -> None:
        super().__init__(viz_prefix, viz_web_port, conf, output_viz_file)
        self.cameras_to_plot = ["camera-rgb"]

    def plot_cubercnn_img(
        self,
        cubercnn_img: torch.Tensor,
        timestamp_ns: int,
    ) -> None:
        # Setting timestamp
        rr.set_time_seconds("frame_time_s", timestamp_ns * 1e-9)

        # Plot image
        # BGR - > RGB, CHW -> HWC
        image = cubercnn_img[[2, 1, 0], :, :].detach().cpu().permute(1, 2, 0).numpy()
        rr.log(
            "camera-rgb_image",
            rr.Image(image),
        )

    def plot_cubercnn_dict(
        self,
        cubercnn_dict: Dict,
        timestamp_ns: int,
        plot_color: List[int],
        suffix: str = "",
    ) -> None:
        # Convert cubercnn GT dict to ATEK format
        atek_format_gt_dict = CubeRCNNModelAdaptor.cubercnn_gt_to_atek_gt(
            cubercnn_dict=cubercnn_dict,
            T_world_camera_np=cubercnn_dict["T_world_camera"],
            camera_label="camera-rgb",
        )

        # Apply confidence filtering
        atek_format_gt_dict["obb3_gt"] = filter_obbs_by_confidence(
            atek_format_gt_dict["obb3_gt"],
            confidence_score=atek_format_gt_dict["scores"],
            confidence_lower_threshold=self.conf.obb_viz.confidence_lower_threshold,
        )
        atek_format_gt_dict["obb2_gt"] = filter_obbs_by_confidence(
            atek_format_gt_dict["obb2_gt"],
            confidence_score=atek_format_gt_dict["scores"],
            confidence_lower_threshold=self.conf.obb_viz.confidence_lower_threshold,
        )

        self.plot_obb3_gt(
            atek_format_gt_dict["obb3_gt"],
            timestamp_ns=timestamp_ns,
            plot_color=plot_color,
            suffix=suffix,
        )

        self.plot_obb2_gt(
            atek_format_gt_dict["obb2_gt"],
            timestamp_ns=timestamp_ns,
            plot_color=plot_color,
            suffix=suffix,
        )

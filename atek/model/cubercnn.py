# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
from argparse import Namespace
from typing import Dict, List

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, get_cfg
from detectron2.engine import default_setup


def create_cubercnn_config(args: Namespace) -> Dict:
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    # add extra configs for data
    cfg.MAX_TRAINING_ATTEMPTS = 3
    cfg.TRAIN_LIST = ""
    cfg.TEST_LIST = ""
    cfg.ID_MAP_JSON = ""
    cfg.OBJ_PROP_JSON = ""
    cfg.CATEGORY_JSON = ""
    cfg.SOLVER.VAL_MAX_ITER = 0

    cfg.merge_from_file(args.config_file)
    if "opts" in args:
        cfg.merge_from_list(args.opts)

    cfg.freeze()
    default_setup(cfg, args)

    model_config = {
        "model_name": args.model_name,
        "ckpt_dir": os.path.dirname(args.config_file),
        "cubercnn_cfg": cfg,
        "post_processor": {
            "score_threshold": args.threshold,
            "category_names": cfg.DATASETS.CATEGORY_NAMES,
        },
    }

    return model_config


class CubercnnPredictionConverter:
    """
    Convert CubeRCNN model predictions from detectron2 instance to a list of dicts

    Args:
        config (Dict): configs need for the conversion, such as score_threshold
    """

    def __init__(self, config: Dict):
        self.config = config

    def __call__(self, model_input: List[Dict], model_prediction: List[List[Dict]]):
        """
        Converts per-frame CubeRCNN prediction to list of dicts
        """
        converted_model_predictions = []

        for input, prediction in zip(model_input, model_prediction):
            dets = prediction["instances"]
            preds_per_frame = []

            if len(dets) == 0:
                continue

            for (
                corners3D,
                center_cam,
                center_2D,
                dimensions,
                bbox_2D,
                pose,
                score,
                scores_full,
                cat_idx,
            ) in zip(
                dets.pred_bbox3D,
                dets.pred_center_cam,
                dets.pred_center_2D,
                dets.pred_dimensions,
                dets.pred_boxes,
                dets.pred_pose,
                dets.scores,
                dets.scores_full,
                dets.pred_classes,
            ):
                if score < self.config["score_threshold"]:
                    continue
                cat = self.config["category_names"][cat_idx]
                predictions_dict = {
                    "sequence_name": input["sequence_name"],
                    "frame_id": input["frame_id"],
                    "timestamp_ns": input["timestamp_ns"],
                    "T_world_cam": input["T_world_camera"],
                    # CubeRCNN dimensions are in reversed order of Aria data convention
                    "dimensions": dimensions.tolist()[::-1],
                    "t_cam_obj": center_cam.tolist(),
                    "R_cam_obj": pose.tolist(),
                    "corners3D": corners3D.tolist(),
                    "center_2D": center_2D.tolist(),
                    "bbox_2D": bbox_2D.tolist(),
                    "score": score.detach().item(),
                    "scores_full": scores_full.tolist(),
                    "category_idx": cat_idx.detach().item(),
                    "category": cat,
                }
                preds_per_frame.append(predictions_dict)

            converted_model_predictions.append(preds_per_frame)

        return converted_model_predictions


class CubercnnInferModel:
    def __init__(
        self,
        model_config: Dict,
    ):
        self.post_processor = CubercnnPredictionConverter(
            model_config["post_processor"]
        )

        self.model = build_model(model_config["cubercnn_cfg"], priors=None)
        _ = DetectionCheckpointer(
            self.model, save_dir=model_config["ckpt_dir"]
        ).resume_or_load(model_config["cubercnn_cfg"].MODEL.WEIGHTS, resume=True)

        self.model.eval()

    def __call__(self, model_input: List[Dict]):
        prediction = self.model(model_input)
        if self.post_processor is None:
            return prediction
        else:
            return self.post_processor(model_input, prediction)


def create_cubercnn_inference_model(args: Namespace):
    """
    Build CubeRCNN model from args
    """
    model_config = create_cubercnn_config(args)
    model = CubercnnInferModel(model_config)

    return model_config, model

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import os
from argparse import Namespace
from typing import Dict, List

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.meta_arch import build_model, RCNN3D  # noqa
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup


def create_cubercnn_config(args: Namespace) -> Dict:
    """
    Create configs and perform basic setups.
    """
    assert args.model_name == "cubercnn"
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
        "ckpt_dir": os.path.dirname(args.config_file),
        "cfg": cfg,
    }

    return model_config


class CubercnnInferModel:
    def __init__(
        self,
        model_config: Dict,
    ):
        self.model = build_model(model_config["cfg"], priors=None)
        _ = DetectionCheckpointer(
            self.model, save_dir=model_config["ckpt_dir"]
        ).resume_or_load(model_config["cfg"].MODEL.WEIGHTS, resume=True)

        self.model.eval()

    def __call__(self, model_input: List[Dict]):
        prediction = self.model(model_input)
        return prediction

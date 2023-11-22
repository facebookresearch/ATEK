# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone  # noqa
from cubercnn.modeling.proposal_generator import RPNWithIgnore  # noqa
from cubercnn.modeling.roi_heads import ROIHeads3D  # noqa


def build_model_with_priors(cfg, priors=None):
    model = build_model(cfg, priors=priors)
    return model

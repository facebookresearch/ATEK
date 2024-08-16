# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import logging
import os
import random
from typing import Dict, Optional

import numpy as np
import torch

import trimesh

from atek.evaluation.surface_reconstruction.surface_reconstruction_utils import (
    compute_pts_to_mesh_dist,
    correct_adt_mesh_gravity,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def evaluate_single_mesh_pair(
    pred_mesh_filename: str,
    gt_mesh_filename: str,
    correct_mesh_gravity: bool = False,
    threshold: float = 0.05,
    sample_num: int = 10000,
    step: int = 50000,
    cut_height: Optional[float] = None,
    rnd_seed: int = 42,
):
    """
    Eval point to faces distance using `point_to_closest_tri_dist`.
    """
    # Setting random seeds
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)

    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    # For testing only
    dev = "cpu"
    logger.info(f"==> [eval_single_mesh_pair] use device {dev}")

    # Load meshes
    gt_mesh = trimesh.load(gt_mesh_filename)
    pred_mesh = trimesh.load(pred_mesh_filename)
    # correct mesh gravity direction, for ADT dataset
    if correct_mesh_gravity:
        gt_mesh = correct_adt_mesh_gravity(gt_mesh)

    # cut the meshes to a plane if needed
    if cut_height is not None:
        cutting_plane = [[0, 0, -1], [0, 0, cut_height]]
        gt_mesh = gt_mesh.slice_plane(
            plane_origin=cutting_plane[1], plane_normal=cutting_plane[0]
        )
        pred_mesh = pred_mesh.slice_plane(
            plane_origin=cutting_plane[1], plane_normal=cutting_plane[0]
        )

    # convert vertices and faces to torch tensors
    pred_vertices = torch.from_numpy(pred_mesh.vertices.view(np.ndarray)).to(dev)
    gt_vertices = torch.from_numpy(gt_mesh.vertices.view(np.ndarray)).to(dev)
    pred_faces = torch.from_numpy(pred_mesh.faces.view(np.ndarray)).to(dev)
    gt_faces = torch.from_numpy(gt_mesh.faces.view(np.ndarray)).to(dev)
    logger.info(f"gt vertices and faces {gt_vertices.shape}, {gt_faces.shape}")
    logger.info(f"pred vertices and faces {pred_vertices.shape}, {pred_faces.shape}")

    # Metric 1: Compute accuracy (from sampled point in pred, to GT)
    pred_pts, _ = trimesh.sample.sample_surface(pred_mesh, sample_num, seed=rnd_seed)
    pred_pts = torch.from_numpy(pred_pts.view(np.ndarray)).to(dev)
    accuracy = compute_pts_to_mesh_dist(pred_pts, gt_faces, gt_vertices, step)

    # Metric 2: completeness
    gt_pts, _ = trimesh.sample.sample_surface(gt_mesh, sample_num, seed=rnd_seed)
    gt_pts = torch.from_numpy(gt_pts.view(np.ndarray)).to(dev)
    completeness = compute_pts_to_mesh_dist(gt_pts, pred_faces, pred_vertices, step)

    # Compute precision and recall based on accuracy and completeness
    precision_perc5 = np.mean((accuracy < 0.05).astype("float"))
    recall_perc5 = np.mean((completeness < 0.05).astype("float"))
    fscore_perc5 = 2 * precision_perc5 * recall_perc5 / (precision_perc5 + recall_perc5)

    # sort to get percentile numbers.
    metrics = {
        "Accuracy_mean_meters": np.mean(accuracy),
        "Completeness_mean_meters": np.mean(completeness),
        "prec@0.05": precision_perc5,
        "recal@0.05": recall_perc5,
        "fscore@0.05": fscore_perc5,
    }

    return metrics, accuracy, completeness



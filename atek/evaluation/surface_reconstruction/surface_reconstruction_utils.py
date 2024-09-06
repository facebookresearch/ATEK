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

import numpy as np

import torch
import trimesh

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def correct_adt_mesh_gravity(mesh: trimesh.Trimesh):
    """
    Change gravity direction of ADT mesh (0, -1, 0), to be consistent with VIO convention (0, 0, -1)
    """
    T_vioWorld_adtWorld = np.array(
        [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    )
    logger.info("Aligning ADT gravity convention to VIO convention.")
    mesh.apply_transform(T_vioWorld_adtWorld)
    return mesh


def point_to_closest_vertex_dist(pts, verts, tris):
    # pts N 3 float
    # verts M 3 float
    # norms M 3 float
    # tris O 3 int
    assert verts.ndim == 2, f"{verts.shape}"
    assert tris.ndim == 2, f"{tris.shape}"
    assert pts.ndim == 2, f"{pts.shape}"
    v0s = verts[None, tris[:, 0], :]
    v1s = verts[None, tris[:, 1], :]
    v2s = verts[None, tris[:, 2], :]
    pts = pts.unsqueeze(1)
    # compute distance to closest vertex
    vs = torch.cat([v0s, v1s, v2s], 0)
    dist_vs = torch.linalg.norm(vs.unsqueeze(1) - pts.unsqueeze(0), 2.0, -1)  # 3, N, M
    dist_vs = torch.min(dist_vs, 0)[0]
    dist_vs = torch.min(dist_vs, 1)[0]  # N
    return dist_vs


def point_to_closest_tri_dist(pts, verts, tris):
    """
    Compute the min distance of points to triangles. If a point doesn't intersect with any triangles
    return a big number (1e6) for that point.
    """
    assert verts.ndim == 2, f"{verts.shape}"
    assert tris.ndim == 2, f"{tris.shape}"
    assert pts.ndim == 2, f"{pts.shape}"

    def dot(a, b):
        return (a * b).sum(-1, keepdim=True)

    # pts N 3 float
    # verts M 3 float
    # norms M 3 float
    # tris O 3 int
    v0s = verts[None, tris[:, 0], :]
    v1s = verts[None, tris[:, 1], :]
    v2s = verts[None, tris[:, 2], :]
    pts = pts.unsqueeze(1)

    # compute if point projects inside triangle
    # https://gamedev.stackexchange.com/questions/28781/easy-way-to-project-point-onto-triangle-or-plane/152476#152476
    u = v1s - v0s
    v = v2s - v0s
    n = torch.cross(u, v)
    w = pts - v0s
    nSq = dot(n, n)
    gamma = dot(torch.cross(u, w, -1), n) / nSq
    beta = dot(torch.cross(w, v, -1), n) / nSq
    alpha = 1.0 - gamma - beta
    valid_alpha = torch.logical_and(0.0 <= alpha, alpha <= 1.0)
    valid_beta = torch.logical_and(0.0 <= beta, beta <= 1.0)
    valid_gamma = torch.logical_and(0.0 <= gamma, gamma <= 1.0)
    projs_to_tri = torch.logical_and(valid_alpha, valid_beta)
    projs_to_tri = torch.logical_and(projs_to_tri, valid_gamma)
    num_proj = projs_to_tri.count_nonzero(1)
    projs_to_tri = projs_to_tri.squeeze(-1)

    # compute distance to triangle plane
    # https://mathworld.wolfram.com/Point-PlaneDistance.html
    n = n / torch.sqrt(nSq)
    dist_tri = dot(n, w).squeeze(-1).abs()
    # set distance to large for point-triangle combinations that do not project
    dist_tri[~projs_to_tri] = 1e6

    dist_tri = torch.min(dist_tri, 1)[0]  # N
    num_proj = num_proj.squeeze(-1)

    return dist_tri, num_proj


def compute_pts_to_mesh_dist(
    pts: torch.Tensor, faces: torch.Tensor, verts: torch.Tensor, step: int
) -> torch.Tensor:
    dev = pts.device
    N = pts.shape[0]
    err = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    dist_tri = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    dist_ver = torch.from_numpy(np.array(N, np.finfo(np.float32).max)).to(dev)
    num_proj = torch.zeros(N).to(dev)
    for i in range(0, faces.shape[0], step):
        dist_tri_i, num_proj_i = point_to_closest_tri_dist(
            pts, verts, faces[i : i + step]
        )
        dist_ver_i = point_to_closest_vertex_dist(pts, verts, faces[i : i + step])
        dist_tri = torch.min(dist_tri_i, dist_tri)
        dist_ver = torch.min(dist_ver_i, dist_ver)
        num_proj = num_proj + num_proj_i

        prog_perc = min((i + step) / faces.shape[0] * 100, 100)
        logger.info(f"Computing pts to mesh progress: {prog_perc:.01f}% \n")
    err = torch.where(num_proj == 0, dist_ver, dist_tri)
    err = err.detach().cpu().numpy()
    return err

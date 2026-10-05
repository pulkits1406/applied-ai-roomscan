"""Point/depth normalisation: per-frame world points with camera-facing normals.

Normals are computed on the depth grid and oriented toward the camera that observed them, so a
fused point keeps which side of a surface was seen (e18: without orientation, furniture backs and
the far faces of shared walls compete with the true wall face).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d

from roomscan.stray import StrayCapture


@dataclass(frozen=True)
class Selection:
    """Which depth samples a stage may use."""
    max_depth: float = 4.5     # m
    min_cos: float = 0.15      # |cos(incidence)|; rejects grazing views and depth-edge normals
    stride: int = 2


ALL = Selection()
CLOSE = Selection(max_depth=2.5, min_cos=0.5)   # wall positions (e18 V5/V6)


def frame_points_normals(cap: StrayCapture, T: np.ndarray, i: int, sel: Selection = ALL):
    d = cap.depth_m(i)[::sel.stride, ::sel.stride]
    c = cap.confidence(i)[::sel.stride, ::sel.stride]
    K = cap.K_depth(i)
    K[:2] /= sel.stride
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    X = np.stack([(u - K[0, 2]) * d / K[0, 0], (v - K[1, 2]) * d / K[1, 1], d], -1)
    dx = np.zeros_like(X)
    dy = np.zeros_like(X)
    dx[:, 1:-1] = X[:, 2:] - X[:, :-2]
    dy[1:-1] = X[2:] - X[:-2]
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-12)
    n[(n * X).sum(-1) > 0] *= -1                     # face the camera
    cos = np.abs((n * X).sum(-1)) / np.maximum(np.linalg.norm(X, axis=-1), 1e-9)
    ok = (c == 2) & (d < sel.max_depth) & (nn[..., 0] > 0) & (cos > sel.min_cos)
    ok[0] = ok[-1] = ok[:, 0] = ok[:, -1] = False
    R, t = T[i, :3, :3], T[i, :3, 3]
    return X[ok] @ R.T + t, n[ok] @ R.T


def fuse(cap: StrayCapture, T: np.ndarray, rows, sel: Selection = ALL, voxel: float = 0.02) -> o3d.geometry.PointCloud:
    P, N = [], []
    for i in rows:
        p, n = frame_points_normals(cap, T, i, sel)
        P.append(p)
        N.append(n)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(P) if P else np.zeros((0, 3))))
    pc.normals = o3d.utility.Vector3dVector(np.concatenate(N) if N else np.zeros((0, 3)))
    pc = pc.voxel_down_sample(voxel)                 # averages normals per voxel
    Nn = np.asarray(pc.normals)
    pc.normals = o3d.utility.Vector3dVector(Nn / np.maximum(np.linalg.norm(Nn, axis=1, keepdims=True), 1e-9))
    return pc

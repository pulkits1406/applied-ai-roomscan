"""Metric, gravity-aligned room point cloud from one room's photos.

Scale (e21, PSEUDO-GT on one capture): the two model families err differently, so the room scale is
the geometric mean of MoGe-2 (EXIF field of view) and MapAnything raw depth x C_MA, where C_MA is
the constant MapAnything is short by (it assumes a ~70 deg FOV; e16). Equivalently, MapAnything
units are multiplied by sqrt(C_MA * r) with r = median(MoGe depth / MapAnything depth) over all
pixels both models predict. e21: 5.3 % median / 14.0 % p90 scale error, no run beyond 25 %,
fixed +-14.3 % interval -> 85 % coverage (leave-one-room-out). C_MA and the interval width were
learned against LiDAR pseudo-GT on the supplied capture only; they are provisional until laser GT.

Geometry: camera poses come from MapAnything's joint reconstruction (the only cross-photo
registration available without poses); per-photo geometry from either model ("ma" | "moge",
chosen in e23).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roomscan.cloudroom import gravity_rotation, point_normals, voxel

C_MA = 1.455                      # e21: median LiDAR / MapAnything-raw depth ratio, all rooms (LORO values 1.433-1.476)
SCALE_HALFWIDTH_90 = 0.143        # e21: fixed relative half-width with 85 % LORO coverage at nominal 90 %
SCALE_SIGMA_REL = SCALE_HALFWIDTH_90 / 1.645


@dataclass
class PhotoCloud:
    P: np.ndarray                 # (N, 3) gravity-aligned (+y up), metres
    N: np.ndarray
    cams: np.ndarray              # (C, 3) camera centres, same frame
    scale_ma_to_m: float          # sqrt(C_MA * r)
    r_moge_over_ma: float
    scale_disagreement: float     # log(MoGe / (MA * C_MA)) for the record; e21: does not predict failures


def _backproject(depth, K):
    H, W = depth.shape
    v, u = np.mgrid[0:H, 0:W]
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def _moge_z_on(ma_hw, pts):
    z = np.where(np.isfinite(pts[..., 2]), pts[..., 2], 0).astype(np.float32)
    ok = cv2.resize(np.isfinite(pts[..., 2]).astype(np.float32), ma_hw[::-1], interpolation=cv2.INTER_AREA) > 0.99
    return np.where(ok, cv2.resize(z, ma_hw[::-1], interpolation=cv2.INTER_AREA), np.nan)


def room_cloud(moge: list[dict], ma: dict, mode: str = "moge", stride: int = 2) -> PhotoCloud:
    depth, mask, pose, K = ma["depth"], ma["mask"], ma["pose"], ma["K"]
    hw = depth.shape[1:]
    num, den = [], []
    for i, mo in enumerate(moge):
        mz = _moge_z_on(hw, mo["points"])
        v = mask[i] & np.isfinite(mz) & (depth[i] > 0)
        num.append(mz[v]); den.append(depth[i][v])
    r = float(np.median(np.concatenate(num) / np.concatenate(den)))
    s = float(np.sqrt(C_MA * r))
    P, N = [], []
    for i, mo in enumerate(moge):
        R, t = pose[i][:3, :3], pose[i][:3, 3] * s
        if mode == "ma":
            X = _backproject(depth[i], K[i]) * s
            valid = mask[i] & (depth[i] > 0)
        else:   # MoGe camera-frame points brought to the ensemble scale (MoGe = r * MA units)
            X = mo["points"] * (s / r)
            valid = mo["mask"]
        p, n = point_normals(X[::stride, ::stride], valid[::stride, ::stride])
        P.append(p @ R.T + t); N.append(n @ R.T)
    P, N = np.concatenate(P), np.concatenate(N)
    G = gravity_rotation(pose[:, :3, :3], N)
    P, N = voxel(P @ G.T, N @ G.T)
    return PhotoCloud(P, N, (pose[:, :3, 3] * s) @ G.T, s, r, float(np.log(r / C_MA)))

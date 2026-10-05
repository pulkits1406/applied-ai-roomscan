"""Synthetic Stray Scanner export of a two-room flat with exactly known geometry (smoke tests).

Geometry (world = Stray/ARKit: metres, +y up): room A interior x 0..4.0, z 0..3.0; room B interior
x 4.1..6.6, z 0..3.0 (a 0.1 m partition between them); floor y = 0, ceiling y = 2.5; a doorway in
the partition, z 1.1..1.9 (0.8 m clear), y 0..2.0 (lintel above). Depth is ray-cast exactly against
the solid blocks; poses follow the verified Stray convention (T_wc, OpenCV camera axes). The camera
turns three times on the spot in each room (tilted down 30 deg, level, tilted up 30 deg, so floor,
upper walls and ceiling are seen, as the capture protocol asks), walking through the doorway between them. No RGB video is
written: the LiDAR tier does not read it.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

W, H = 256, 192                 # depth map size of the supplied exports (write(..., depth_wh) overrides)
RGB_K = np.array([[1580.0, 0, 959.5], [0, 1580.0, 719.5], [0, 0, 1]])
FPS = 30.0
CEILING = 2.5
# solid blocks (xmin, ymin, zmin, xmax, ymax, zmax)
BLOCKS = [
    (-0.2, -0.2, -0.2, 6.8, 0.0, 3.2),          # floor slab
    (-0.2, CEILING, -0.2, 6.8, CEILING + 0.2, 3.2),
    (-0.2, 0.0, -0.2, 0.0, CEILING, 3.2),       # west wall
    (6.6, 0.0, -0.2, 6.8, CEILING, 3.2),        # east wall
    (-0.2, 0.0, -0.2, 6.8, CEILING, 0.0),       # south wall
    (-0.2, 0.0, 3.0, 6.8, CEILING, 3.2),        # north wall
    (4.0, 0.0, 0.0, 4.1, CEILING, 1.1),         # partition south of the doorway
    (4.0, 0.0, 1.9, 4.1, CEILING, 3.0),         # partition north of the doorway
    (4.0, 2.0, 1.1, 4.1, CEILING, 1.9),         # lintel
]
TRUTH = {"A": {"walls": [4.0, 3.0, 4.0, 3.0], "ceiling": CEILING, "area": 12.0},
         "B": {"walls": [2.5, 3.0, 2.5, 3.0], "ceiling": CEILING, "area": 7.5},
         "doorway_width": 0.8}


def _R(yaw, pitch):
    f = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
    right = np.cross(f, [0.0, 1.0, 0.0]); right /= np.linalg.norm(right)
    down = np.cross(f, right)
    return np.stack([right, down, f], 1)                # columns = camera x, y, z in world


def trajectory(fps: float = FPS):
    T = []

    def spin(c, pitch, secs=6.0):
        for k in range(int(secs * fps)):
            M = np.eye(4); M[:3, :3] = _R(2 * np.pi * k / (secs * fps), np.radians(pitch)); M[:3, 3] = c
            T.append(M)

    def walk(a, b, yaw, secs=2.0):
        for k in range(int(secs * fps)):
            M = np.eye(4); M[:3, :3] = _R(yaw, 0.0); M[:3, 3] = a + (b - a) * k / (secs * fps)
            T.append(M)

    A, B = np.array([2.0, 1.4, 1.5]), np.array([5.35, 1.4, 1.5])
    spin(A, -30.0); spin(A, 0.0); spin(A, 30.0); walk(A, B, np.pi / 2); spin(B, -30.0); spin(B, 0.0); spin(B, 30.0)
    return T


def depth(T, wh=(W, H)):
    w, h = wh
    Kd = RGB_K.copy(); Kd[0] *= w / 1920; Kd[1] *= h / 1440
    v, u = np.mgrid[0:h, 0:w]
    dc = np.stack([(u + 0.5 - Kd[0, 2]) / Kd[0, 0], (v + 0.5 - Kd[1, 2]) / Kd[1, 1], np.ones((h, w))], -1).reshape(-1, 3)
    dw = dc @ T[:3, :3].T
    o = T[:3, 3]
    best = np.full(len(dw), np.inf)
    inv = 1.0 / np.where(np.abs(dw) < 1e-12, 1e-12, dw)
    for b in BLOCKS:
        lo, hi = (np.array(b[:3]) - o) * inv, (np.array(b[3:]) - o) * inv
        tmin, tmax = np.minimum(lo, hi).max(1), np.maximum(lo, hi).min(1)
        hit = (tmax >= tmin) & (tmax > 0) & (tmin > 0)
        best = np.where(hit & (tmin < best), tmin, best)
    return best.reshape(h, w)                          # dc has z = 1, so t is the depth


def write(out: str | Path, fps: float = FPS, depth_wh: tuple[int, int] = (W, H)) -> Path:
    out = Path(out)
    (out / "depth").mkdir(parents=True, exist_ok=True); (out / "confidence").mkdir(exist_ok=True)
    rows = []
    for i, T in enumerate(trajectory(fps)):
        d = depth(T, depth_wh)
        cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), np.clip(np.nan_to_num(d * 1000, posinf=0), 0, 65535).astype(np.uint16))
        cv2.imwrite(str(out / "confidence" / f"{i:06d}.png"), np.where(np.isfinite(d) & (d < 6), 2, 0).astype(np.uint8))
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        rows.append(f"{i / fps:.6f}, {i:06d}, {T[0, 3]:.6f}, {T[1, 3]:.6f}, {T[2, 3]:.6f}, {q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}, "
                    f"{RGB_K[0, 0]}, {RGB_K[1, 1]}, {RGB_K[0, 2]}, {RGB_K[1, 2]}, , ")
    (out / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y\n"
                                      + "\n".join(rows) + "\n")
    (out / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
    return out

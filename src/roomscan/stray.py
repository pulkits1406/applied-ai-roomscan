"""Loader for Stray Scanner exports (rgb.mp4, depth/, confidence/, odometry.csv, imu.csv).

Conventions (verified empirically in experiments/e01_pose_convention):
- depth PNG: uint16 millimetres, 256x192, aligned with the 1920x1440 RGB frame (same 4:3 FOV).
- confidence PNG: uint8 in {0, 1, 2} (ARKit low / medium / high).
- odometry row i corresponds to depth/confidence frame i, and to decoded video frame i - 1:
  the encoded video lacks the first frame (container declares N frames, N - 1 decode; the
  video PTS gap pattern matches odometry rows 1..N-1 exactly, and RGB/depth edge correlation
  peaks at that pairing). Row 0 has no RGB.
- pose (x, y, z, qx, qy, qz, qw) is camera-to-world with an OpenCV-convention camera frame
  (x right, y down, z forward); applying the ARKit camera flip on top is wrong (200-300 mm
  reprojection residual vs 5-8 mm). Per-row fx, fy, cx, cy are for the 1920x1440 RGB image
  and vary frame to frame (autofocus).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

RGB_WH = (1920, 1440)
VIDEO_ROW_OFFSET = 1  # decoded video frame j is odometry/depth row j + 1
# ARKit camera axes (x right, y up, z back) -> OpenCV camera axes (x right, y down, z forward).
# Not applied to Stray poses, which are already OpenCV-convention; kept for ARKit-native inputs.
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0])


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


@dataclass
class StrayCapture:
    root: Path
    odometry: pd.DataFrame
    imu: pd.DataFrame

    @classmethod
    def load(cls, root: str | Path) -> "StrayCapture":
        root = Path(root)
        return cls(root, _read_csv(root / "odometry.csv"), _read_csv(root / "imu.csv"))

    def __len__(self) -> int:
        return len(self.odometry)

    @property
    def timestamps(self) -> np.ndarray:
        return self.odometry["timestamp"].to_numpy()

    def T_wc(self, i: int | None = None) -> np.ndarray:
        """Camera-to-world 4x4 with an OpenCV-convention camera frame. Shape (4,4) or (N,4,4)."""
        o = self.odometry if i is None else self.odometry.iloc[[i]]
        R = Rotation.from_quat(o[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
        T = np.tile(np.eye(4), (len(o), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = o[["x", "y", "z"]].to_numpy()
        return T[0] if i is not None else T

    def K_rgb(self, i: int) -> np.ndarray:
        r = self.odometry.iloc[i]
        return np.array([[r.fx, 0, r.cx], [0, r.fy, r.cy], [0, 0, 1.0]])

    def K_depth(self, i: int, depth_wh: tuple[int, int] = (256, 192)) -> np.ndarray:
        K = self.K_rgb(i).copy()
        K[0] *= depth_wh[0] / RGB_WH[0]
        K[1] *= depth_wh[1] / RGB_WH[1]
        return K

    def iter_rgb(self, rows: set[int] | None = None):
        """Yield (odometry_row, BGR image) by sequential decode; seeking HEVC is not frame-exact."""
        cap = cv2.VideoCapture(str(self.root / "rgb.mp4"))
        try:
            j = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                row = j + VIDEO_ROW_OFFSET
                if rows is None or row in rows:
                    yield row, frame
                if rows is not None and row >= max(rows):
                    break
                j += 1
        finally:
            cap.release()

    def depth_m(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.root / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0

    def confidence(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.root / "confidence" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)

    def points_world(self, i: int, min_conf: int = 2, max_depth: float = 5.0, stride: int = 1) -> np.ndarray:
        """Back-project frame i's depth to world coordinates (metres)."""
        d = self.depth_m(i)[::stride, ::stride]
        c = self.confidence(i)[::stride, ::stride]
        K = self.K_depth(i)
        K[:2] /= stride
        v, u = np.nonzero((c >= min_conf) & (d > 0) & (d < max_depth))
        z = d[v, u]
        pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
        T = self.T_wc(i)
        return pc @ T[:3, :3].T + T[:3, 3]

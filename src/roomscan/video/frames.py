"""Plain video -> upright keyframes (RGB only; no intrinsics, poses or IMU are read).

Orientation comes from the container's rotation metadata (iPhone Camera .MOV files carry it);
files without it (e.g. Stray Scanner's rgb.mp4, recorded in sensor orientation with the phone
held upright) need an explicit rotation. Keyframes are sampled by timestamp at `fps`, so
variable-frame-rate files are handled.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


@dataclass
class Keyframe:
    index: int
    t: float                   # seconds from the start of the clip
    rgb: np.ndarray            # upright, long side <= max_side


def read(path: str | Path, fps: float = 2.0, rotate: str | int = "auto", max_side: int = 960) -> tuple[list[Keyframe], dict]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video {path}")
    meta_rot = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) if hasattr(cv2, "CAP_PROP_ORIENTATION_META") else 0
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)          # rotate explicitly, once, below
    rot = meta_rot if rotate == "auto" else int(rotate)
    info = {"rotation_deg": rot, "rotation_source": "metadata" if rotate == "auto" else "argument", "fps_sampled": fps}
    out, nxt, k = [], 0.0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 < nxt:
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            break
        s = min(1.0, max_side / max(bgr.shape[:2]))
        if s < 1:
            bgr = cv2.resize(bgr, (round(bgr.shape[1] * s), round(bgr.shape[0] * s)), interpolation=cv2.INTER_AREA)
        if ROT.get(rot % 360) is not None:
            bgr = cv2.rotate(bgr, ROT[rot % 360])
        out.append(Keyframe(k, t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        k += 1
        nxt = t + 1.0 / fps
    cap.release()
    info["n_keyframes"] = len(out)
    info["duration_s"] = round(out[-1].t - out[0].t, 2) if out else 0.0
    return out, info

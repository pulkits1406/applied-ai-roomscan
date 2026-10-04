"""Shared helpers for e13: MapAnything inference on photo-like frames vs LiDAR pseudo-GT.

Conventions:
- Cached RGB frames are 960x720 sensor-landscape; every frame in this capture needs
  cv2.ROTATE_90_CLOCKWISE to be upright (720x960). Landscape pixel (u, v) maps to upright
  (H-1-v, u), so upright K has fx'=fy, fy'=fx, cx'=H-1-cy, cy'=cx (H=720).
- Model input resolution is 378x504 (MapAnything resolution set 504, aspect 0.750), which is
  exactly 720x960 scaled by 0.525, so preprocessing never crops and the prediction maps back
  to the full image by a pure resize.
- Z-depth is invariant to the in-plane camera rotation, and camera centres are invariant to
  it too, so landscape LiDAR depth and ARKit centres compare directly with upright predictions.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAP = "single_scan_with_ceiling"
FRAMES = ROOT / f"cache/frames/{CAP}"
OUT = ROOT / "experiments/e13_multiview_photo/out"
HF_ID = "facebook/map-anything-apache"
RES_SET = 504
LIDAR_MAX = 6.0  # same cut as the e06 MoGe-2 baseline

os.environ.setdefault("HF_HOME", str(ROOT / "cache/e13/hf"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

META = {m["row"]: m for m in json.loads((FRAMES / "frames.json").read_text())["frames"]}


def upright_rgb(row: int) -> np.ndarray:
    bgr = cv2.imread(str(FRAMES / "rgb" / f"{row:06d}.jpg"))
    return cv2.rotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)


def upright_K(row: int, H_land: int = 720) -> np.ndarray:
    K = np.array(META[row]["K"], dtype=np.float64)
    return np.array([[K[1, 1], 0, H_land - 1 - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]])


def centre(row: int) -> np.ndarray:
    return np.array(META[row]["T_wc"])[:3, 3]


# Upright camera axes in landscape camera coords: x' = -y, y' = x, z' = z.
R_LAND_UP = np.array([[0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])


def R_wc_upright(row: int) -> np.ndarray:
    return np.array(META[row]["T_wc"])[:3, :3] @ R_LAND_UP


def load_model(device: str):
    import torch
    from mapanything.models import MapAnything
    return MapAnything.from_pretrained(HF_ID).to(torch.device(device)).eval()


def run_model(model, rows: list[int], with_K: bool, amp: str = "bf16") -> dict:
    """Run MapAnything on the given rows; return per-view depth (upright, model res), centres, K, secs."""
    import torch
    from mapanything.utils.image import preprocess_inputs
    views = []
    for r in rows:
        v = {"img": upright_rgb(r)}
        if with_K:
            v["intrinsics"] = torch.tensor(upright_K(r), dtype=torch.float32)
        views.append(v)
    t0 = time.time()
    pv = preprocess_inputs(views, resolution_set=RES_SET)
    with torch.no_grad():
        preds = model.infer(pv, memory_efficient_inference=False, use_amp=amp != "fp32", amp_dtype=amp,
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    if model.device.type == "mps":
        torch.mps.synchronize()
    sec = time.time() - t0
    out = {"depth": [], "mask": [], "centre": [], "R": [], "K": [], "sec": sec}
    for p in preds:
        out["depth"].append(p["depth_z"][0, ..., 0].float().cpu().numpy())
        out["mask"].append(p["mask"][0, ..., 0].cpu().numpy() > 0.5)
        out["centre"].append(p["camera_poses"][0, :3, 3].float().cpu().numpy())
        out["R"].append(p["camera_poses"][0, :3, :3].float().cpu().numpy())
        out["K"].append(p["intrinsics"][0].float().cpu().numpy())
    return out


_lidar_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def lidar(row: int):
    if row not in _lidar_cache:
        root = ROOT / CAP
        d = cv2.imread(str(root / "depth" / f"{row:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        c = cv2.imread(str(root / "confidence" / f"{row:06d}.png"), cv2.IMREAD_UNCHANGED)
        _lidar_cache[row] = (d, (c == 2) & (d > 0) & (d < LIDAR_MAX))
    return _lidar_cache[row]


def pred_vs_lidar(depth_up: np.ndarray, mask_up: np.ndarray, row: int):
    """Return (pred, lidar) z-depth samples on valid confidence-2 pixels of the 256x192 LiDAR grid."""
    d = np.where(mask_up, depth_up, np.nan).astype(np.float32)
    d = cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE)  # back to sensor landscape
    d = cv2.resize(d, (256, 192), interpolation=cv2.INTER_AREA)
    lid, ok = lidar(row)
    v = ok & np.isfinite(d) & (d > 0)
    return d[v], lid[v]


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Sim(3) s, R, t minimising ||dst - (s R src + t)||^2 (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(xd.T @ xs / len(src))
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    R = U @ D @ Vt
    var_s = (xs ** 2).sum() / len(src)
    s = float(np.trace(np.diag(S) @ D) / var_s)
    return s, R, mu_d - s * R @ mu_s


def spread_ok(c: np.ndarray, min_extent: float = 0.3, min_ratio: float = 0.15) -> bool:
    """Centres usable for Sim(3) scale: spread > min_extent m and not nearly collinear."""
    sv = np.linalg.svd(c - c.mean(0), compute_uv=False) / np.sqrt(len(c))
    return bool(sv[0] > min_extent / 2 and sv[1] / sv[0] > min_ratio)

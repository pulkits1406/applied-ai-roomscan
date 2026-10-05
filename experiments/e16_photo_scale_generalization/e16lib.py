"""Shared helpers for e16 (and e17): MapAnything inference with per-view caching, evaluation from cache.

Reuses e13's conventions (experiments/e13_multiview_photo/common.py): upright 720x960 inputs fed at
378x504 (no crop), LiDAR pseudo-GT = Stray depth with confidence==2 and < 6 m on the 256x192
landscape grid. Cached predictions are stored in landscape 256x192 so they line up with LiDAR and
with the e12 MoGe-2 cache (cache/e12/<row>.npy) pixel for pixel.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e13_multiview_photo"))
import common as c13  # noqa: E402  (sets HF_HOME=cache/e13/hf before any HF import)

META, lidar, umeyama, spread_ok, centre, R_wc_upright = (c13.META, c13.lidar, c13.umeyama, c13.spread_ok,
                                                         c13.centre, c13.R_wc_upright)
upright_rgb, upright_K = c13.upright_rgb, c13.upright_K
SETS = json.loads((ROOT / "experiments/e12_photo_scale/out/photo_sets.json").read_text())
MODEL_SCALE = 0.525  # model input 378x504 = 0.525 x upright 720x960


def mem_ok(max_swap_mb: float = 6000, min_free_pages: int = 50000) -> tuple[bool, str]:
    sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    used = float(sw.split("used = ")[1].split("M")[0])
    vs = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    free = int([l for l in vs.splitlines() if l.startswith("Pages free")][0].split()[-1].rstrip("."))
    return used <= max_swap_mb and free >= min_free_pages, f"swap {used:.0f}MB free_pages {free}"


def wait_for_memory(log=print, max_wait_s: int = 900):
    t0 = time.time()
    while True:
        ok, msg = mem_ok()
        if ok:
            return msg
        log(f"[mem] waiting: {msg}")
        if time.time() - t0 > max_wait_s:
            raise SystemExit(f"memory did not recover in {max_wait_s}s: {msg}")
        time.sleep(10)


def infer_views(model, imgs: list[np.ndarray], Ks: list[np.ndarray] | None, amp: str = "bf16") -> dict:
    """MapAnything on upright RGB images (+ optional upright full-res K). Returns model-res outputs."""
    import torch
    from mapanything.utils.image import preprocess_inputs
    views = []
    for i, im in enumerate(imgs):
        v = {"img": im}
        if Ks is not None:
            v["intrinsics"] = torch.tensor(Ks[i], dtype=torch.float32)
        views.append(v)
    t0 = time.time()
    pv = preprocess_inputs(views, resolution_set=c13.RES_SET)
    with torch.no_grad():
        preds = model.infer(pv, memory_efficient_inference=False, use_amp=amp != "fp32", amp_dtype=amp,
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    if model.device.type == "mps":
        torch.mps.synchronize()
    sec = time.time() - t0
    out = {"depth_up": [], "mask_up": [], "poses": [], "K": [], "sec": sec}
    for p in preds:
        out["depth_up"].append(p["depth_z"][0, ..., 0].float().cpu().numpy())
        out["mask_up"].append(p["mask"][0, ..., 0].cpu().numpy() > 0.5)
        out["poses"].append(p["camera_poses"][0].float().cpu().numpy())
        out["K"].append(p["intrinsics"][0].float().cpu().numpy())
    del preds, pv
    if model.device.type == "mps":
        torch.mps.empty_cache()  # return the allocator cache so free pages recover between runs
    return out


def to_land(depth_up: np.ndarray, mask_up: np.ndarray) -> np.ndarray:
    """Upright model-res depth -> landscape 256x192 grid (NaN where masked); same ops as e13 pred_vs_lidar."""
    d = np.where(mask_up, depth_up, np.nan).astype(np.float32)
    d = cv2.rotate(d, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return cv2.resize(d, (256, 192), interpolation=cv2.INTER_AREA)


def save_npz(path: Path, rows, res: dict, **extra):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=np.array(rows), sec=res["sec"],
                        depth_land=np.stack([to_land(d, m) for d, m in zip(res["depth_up"], res["mask_up"])]).astype(np.float16),
                        K_pred=np.stack(res["K"]), poses=np.stack(res["poses"]), **extra)


def samples(d_land: np.ndarray, row: int):
    lid, ok = lidar(row)
    d = d_land.astype(np.float32)
    v = ok & np.isfinite(d) & (d > 0)
    return d[v], lid[v]


def moge_land(row: int) -> np.ndarray:
    return np.load(ROOT / f"cache/e12/{row:06d}.npy")


def evaluate(rows: list[int], depth_land: np.ndarray, K_pred: np.ndarray, poses: np.ndarray, sec: float) -> dict:
    """Same metrics as e13 run_subsets.evaluate, computed from cached arrays, plus MoGe-2 on the same rows."""
    P, L, per_img = [], [], []
    for i, r in enumerate(rows):
        p, l = samples(depth_land[i], r)
        P.append(p); L.append(l)
        per_img.append(float(np.median(p / l)) if len(p) > 50 else None)
    Pa, La = np.concatenate(P), np.concatenate(L)
    ratio = float(np.median(Pa / La)); s = 1.0 / ratio
    rec = {"rows": [int(r) for r in rows], "K": len(rows), "sec": round(float(sec), 3), "n_px": int(len(Pa)),
           "ratio_pred_over_lidar": ratio, "scale_err_pct": 100 * (ratio - 1), "per_image_ratio": per_img,
           "absrel_aligned": float(np.mean(np.abs(s * Pa - La) / La)),
           "absrel_aligned_median": float(np.median(np.abs(s * Pa - La) / La))}
    fx_true = np.array([META[r]["K"][1][1] * MODEL_SCALE for r in rows])
    fy_true = np.array([META[r]["K"][0][0] * MODEL_SCALE for r in rows])
    fx_pred, fy_pred = K_pred[:, 0, 0].astype(float), K_pred[:, 1, 1].astype(float)
    rec["per_image_focal_ratio"] = (fx_pred / fx_true).tolist()
    rec["per_image_focal_ratio_y"] = (fy_pred / fy_true).tolist()
    rec["focal_err_pct"] = float(100 * np.median(fx_pred / fx_true - 1))
    # Pinhole Z = f*S/s: a mis-estimated focal scales that view's depth by the same factor.
    Pc = np.concatenate([p * (ft / fp) for p, ft, fp in zip(P, fx_true, fx_pred)])
    rec["ratio_focalcorr"] = float(np.median(Pc / La))
    rec["scale_err_pct_focalcorr"] = 100 * (rec["ratio_focalcorr"] - 1)
    sc = 1 / rec["ratio_focalcorr"]
    rec["absrel_aligned_focalcorr"] = float(np.mean(np.abs(sc * Pc - La) / La))
    # MoGe-2 (FOV given), pooled over the same rows and the same confidence-2 pixels definition.
    PM, LM = zip(*[samples(moge_land(r), r) for r in rows])
    rec["moge_ratio"] = float(np.median(np.concatenate(PM) / np.concatenate(LM)))
    rec["moge_err_pct"] = 100 * (rec["moge_ratio"] - 1)
    cp, cg = poses[:, :3, 3].astype(float), np.array([centre(r) for r in rows])
    if len(rows) >= 2:
        iu = np.triu_indices(len(rows), 1)
        bp = np.linalg.norm(cp[:, None] - cp[None], axis=-1)[iu]
        bg = np.linalg.norm(cg[:, None] - cg[None], axis=-1)[iu]
        keep = bg > 0.3
        rec["baseline_ratio_pred_over_gt"] = float(np.median(bp[keep] / bg[keep])) if keep.any() else None
        rec["n_baselines"] = int(keep.sum())
    if len(rows) >= 3 and spread_ok(cg):
        sg, R, t = umeyama(cp, cg)
        rec["sim3_ratio_pred_over_gt"] = 1.0 / sg
        rec["sim3_rmse_m"] = float(np.sqrt(np.mean(np.sum((cg - (sg * cp @ R.T + t)) ** 2, 1))))
    return rec


def load_model(device: str = "mps", bf16_backbone: bool = True):
    """bf16_backbone casts the DINOv2 encoder and the info-sharing transformer (1.14 of 1.25 B params) to
    bf16, saving ~2.3 GB. Those modules already run under bf16 autocast; the geometric-input encoders and
    heads, which the model runs with autocast disabled, stay fp32. Equivalence: out/check_bf16_weights.json."""
    import gc
    import torch
    from mapanything.models import MapAnything
    m = MapAnything.from_pretrained(c13.HF_ID)
    if bf16_backbone:  # cast on CPU before the device copy so the GPU never holds the fp32 backbone
        m.encoder.to(torch.bfloat16); m.info_sharing.to(torch.bfloat16)
    m = m.to(torch.device(device)).eval()
    gc.collect()
    return m

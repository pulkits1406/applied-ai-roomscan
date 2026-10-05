"""MapAnything on sliding windows of keyframes, chained by Sim(3) into one trajectory (cache/e13/.venv).

Usage:
  cache/e13/.venv/bin/python experiments/e20_video_tracking/mapanything_windows.py infer <capture> --stride 3 --win 16 --overlap 6
  cache/e13/.venv/bin/python experiments/e20_video_tracking/mapanything_windows.py chain <capture> --stride 3 --win 16 --overlap 6
infer caches one npz per window (cam-to-world poses, predicted K, 48x64 landscape depth) under
cache/e20/mapanything/<capture>_s<stride>_w<win>_o<overlap>/; chain needs only those files.
Views are upright 720x960 frames with their true upright K (MapAnything still predicts ~70 deg FOV, e16).
Chaining: rotation = SO(3) mean of R_chain R_win^T over the overlap frames; scale = median ratio of the
two windows' depth predictions on the overlap frames (pose and depth share one scale within a window);
translation from the overlap centroids. Frames already placed keep their earlier pose.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE, OUT, ROOT, evaluate_method, img_path, load_frames, mem_ok, save_poses, wait_for_memory  # noqa: E402

HF_ID = "facebook/map-anything-apache"
os.environ.setdefault("HF_HOME", str(ROOT / "cache/e13/hf"))  # e13 download of the Apache checkpoint
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
R_LAND_UP = np.array([[0, 1.0, 0], [-1, 0, 0], [0, 0, 1]])  # upright camera axes in landscape camera coords


def windows(n: int, win: int, overlap: int) -> list[list[int]]:
    step = win - overlap
    ws, s = [], 0
    while True:
        ws.append(list(range(s, min(s + win, n))))
        if s + win >= n:
            break
        s += step
    return ws


def run_dir(capture, stride, win, overlap) -> Path:
    return CACHE / "mapanything" / f"{capture}_s{stride}_w{win}_o{overlap}"


def infer(capture: str, stride: int, win: int, overlap: int, max_minutes: float):
    import cv2
    import torch
    frames = load_frames(capture)[::stride]
    d = run_dir(capture, stride, win, overlap)
    d.mkdir(parents=True, exist_ok=True)
    ws = windows(len(frames), win, overlap)
    todo = [k for k in range(len(ws)) if not (d / f"w{k:04d}.npz").exists()]
    print(f"{len(frames)} frames, {len(ws)} windows, {len(todo)} to do", flush=True)
    if not todo:
        return
    print(wait_for_memory())
    model = load_model("mps")
    t_start = time.time()
    for k in todo:
        rows = [frames[i]["row"] for i in ws[k]]
        imgs, Ks = [], []
        for f in (frames[i] for i in ws[k]):
            bgr = cv2.imread(str(img_path(capture, f["row"])))
            imgs.append(cv2.rotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE))
            K = np.array(f["K"])
            Ks.append(np.array([[K[1, 1], 0, 720 - 1 - K[1, 2]], [0, K[0, 0], K[0, 2]], [0, 0, 1.0]]))
        res = infer_views(model, imgs, Ks)
        dl = []
        for dep, m in zip(res["depth_up"], res["mask_up"]):
            x = np.where(m, dep, np.nan).astype(np.float32)
            dl.append(cv2.resize(cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE), (64, 48), interpolation=cv2.INTER_AREA))
        np.savez(d / f"w{k:04d}.npz", rows=np.array(rows), poses=np.stack(res["poses"]), K_pred=np.stack(res["K"]),
                 depth48=np.stack(dl).astype(np.float16), sec=res["sec"])
        ok, msg = mem_ok()
        print(f"window {k}/{len(ws)} {len(rows)} views {res['sec']:.1f}s {msg}", flush=True)
        if res["sec"] > 180:  # a window normally takes ~20-30 s; this long means the machine is paging
            print("window took > 180 s: stopping (rerun to resume)", flush=True)
            break
        if not ok:
            wait_for_memory(max_wait_s=300)
        if time.time() - t_start > 60 * max_minutes:
            print("time budget reached; rerun to resume", flush=True)
            break
    del model
    torch.mps.empty_cache()


def load_model(device: str = "mps"):
    """As e16lib.load_model: Apache checkpoint, DINOv2 encoder + info-sharing transformer cast to bf16 on CPU
    before the device copy (RSS ~4.5 GB); heads stay fp32."""
    import gc
    import torch
    from mapanything.models import MapAnything
    m = MapAnything.from_pretrained(HF_ID)
    m.encoder.to(torch.bfloat16)
    m.info_sharing.to(torch.bfloat16)
    m = m.to(torch.device(device)).eval()
    gc.collect()
    return m


def infer_views(model, imgs, Ks) -> dict:
    """As e16lib.infer_views (resolution set 504 -> 378x504 upright input, no crop; bf16 autocast), but with
    memory_efficient_inference=True to keep the 12-16 view peak lower."""
    import torch
    from mapanything.utils.image import preprocess_inputs
    views = [{"img": im, "intrinsics": torch.tensor(K, dtype=torch.float32)} for im, K in zip(imgs, Ks)]
    t0 = time.time()
    pv = preprocess_inputs(views, resolution_set=504)
    with torch.no_grad():
        preds = model.infer(pv, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True, apply_confidence_mask=False)
    torch.mps.synchronize()
    out = {"depth_up": [], "mask_up": [], "poses": [], "K": [], "sec": time.time() - t0}
    for p in preds:
        out["depth_up"].append(p["depth_z"][0, ..., 0].float().cpu().numpy())
        out["mask_up"].append(p["mask"][0, ..., 0].cpu().numpy() > 0.5)
        out["poses"].append(p["camera_poses"][0].float().cpu().numpy())
        out["K"].append(p["intrinsics"][0].float().cpu().numpy())
    del preds, pv
    torch.mps.empty_cache()
    return out


def so3_mean(Rs):
    U, _, Vt = np.linalg.svd(sum(Rs))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def chain(capture: str, stride: int, win: int, overlap: int) -> dict:
    d = run_dir(capture, stride, win, overlap)
    files = sorted(d.glob("w*.npz"))
    W = [np.load(f) for f in files]
    C: dict[int, np.ndarray] = {}   # row -> chained cam-to-world (upright camera axes)
    Dc: dict[int, np.ndarray] = {}  # row -> depth in chain units
    link_scales, link_rot_spread, sec = [], [], 0.0
    for k, w in enumerate(W):
        rows, P, dep = [int(r) for r in w["rows"]], w["poses"].astype(np.float64), w["depth48"].astype(np.float32)
        sec += float(w["sec"])
        if k == 0:
            s, R, t = 1.0, np.eye(3), np.zeros(3)
        else:
            ov = [i for i, r in enumerate(rows) if r in C]
            if not ov:
                raise SystemExit(f"window {k} has no overlap with the chain")
            R = so3_mean([C[rows[i]][:3, :3] @ P[i, :3, :3].T for i in ov])
            link_rot_spread.append(max(np.degrees(np.arccos(np.clip(
                (np.trace(R.T @ C[rows[i]][:3, :3] @ P[i, :3, :3].T) - 1) / 2, -1, 1))) for i in ov))
            ratios = []
            for i in ov:
                a, b = Dc[rows[i]], dep[i]
                v = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
                if v.sum() > 50:
                    ratios.append(np.median(a[v] / b[v]))
            s = float(np.median(ratios))
            link_scales.append(s)
            cC = np.array([C[rows[i]][:3, 3] for i in ov])
            cP = P[ov, :3, 3]
            t = (cC - s * cP @ R.T).mean(0)
        for i, r in enumerate(rows):
            if r in C:
                continue
            T = np.eye(4)
            T[:3, :3] = R @ P[i, :3, :3]
            T[:3, 3] = s * R @ P[i, :3, 3] + t
            C[r] = T
            Dc[r] = s * dep[i]
    poses = {}
    for r, T in C.items():
        Tl = T.copy()
        Tl[:3, :3] = T[:3, :3] @ R_LAND_UP.T  # back to the landscape camera of frames.json
        poses[r] = Tl
    ls = np.array(link_scales)
    cum = np.cumprod(ls) if len(ls) else np.array([1.0])
    return poses, {"n_windows": len(W), "sec_inference": sec, "link_scale_median": float(np.median(ls)) if len(ls) else 1.0,
                   "link_scale_abs_dev_median_pct": float(100 * np.median(np.abs(ls - 1))) if len(ls) else 0.0,
                   "cum_scale_minmax": [float(cum.min()), float(cum.max())],
                   "link_rot_spread_deg_median": float(np.median(link_rot_spread)) if link_rot_spread else 0.0,
                   "link_rot_spread_deg_max": float(np.max(link_rot_spread)) if link_rot_spread else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["infer", "chain"])
    ap.add_argument("capture")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--win", type=int, default=16)
    ap.add_argument("--overlap", type=int, default=6)
    ap.add_argument("--max-minutes", type=float, default=18)
    a = ap.parse_args()
    if a.cmd == "infer":
        infer(a.capture, a.stride, a.win, a.overlap, a.max_minutes)
        return
    poses, info = chain(a.capture, a.stride, a.win, a.overlap)
    rows = [f["row"] for f in load_frames(a.capture)[::a.stride]]
    name = f"mapanything_s{a.stride}_w{a.win}_o{a.overlap}"
    save_poses(CACHE / "poses" / f"{a.capture}_{name}.json", poses, None, method=name)
    ev = evaluate_method(a.capture, rows, poses, None, metric=True, extra={"method": name, **info})
    (OUT / f"{a.capture}_{name}_eval.json").write_text(json.dumps(ev, indent=1))
    lg = ev["largest"]
    print(name, json.dumps(info), "ATE", round(lg["ate_rmse_m"], 3), "scale_err%", round(lg["metric_scale_err_pct"], 1),
          "win", lg.get("win_scale_minmax"), "winATE", lg.get("win_ate_median_m"),
          "drift", lg.get("drift_fit30", {}).get("err_end_m"), "LR", lg.get("longrange_dist_err_pct"))


if __name__ == "__main__":
    main()

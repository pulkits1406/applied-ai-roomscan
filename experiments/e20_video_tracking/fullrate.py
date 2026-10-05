"""Does the full 60 fps video bridge the blind keyframe links? (targeted test on the blind intervals only)

Usage:
  uv run python experiments/e20_video_tracking/fullrate.py dump <capture>     # decode rgb.mp4 rows (project venv)
  PYTORCH_ENABLE_MPS_FALLBACK=1 TORCH_HOME=cache/e20/torch cache/e20/.venv/bin/python \
      experiments/e20_video_tracking/fullrate.py match <capture>              # ALIKED+LightGlue r -> r+1
  uv run python experiments/e20_video_tracking/fullrate.py eval <capture>     # chained rotation vs ARKit
An interval spans every video row from the keyframe before a blind link (< 20 matches between
neighbouring cached keyframes, out/<capture>_blind.json) to the keyframe after it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, OUT, ROOT, load_frames, rot_deg  # noqa: E402


def intervals(cap: str) -> list[list[int]]:
    fr = load_frames(cap)
    M = {}
    for p in sorted((CACHE / "match" / cap).glob("chunk_*.npz")):
        z = np.load(p)
        M.update({k: len(z[k]) for k in z.files})
    ivs = []
    for a, b in zip(fr[:-1], fr[1:]):
        if M.get(f"{a['row']}_{b['row']}", 0) < 20:
            if ivs and ivs[-1][1] >= a["row"]:
                ivs[-1][1] = b["row"]
            else:
                ivs.append([a["row"], b["row"]])
    return ivs


def dump(cap: str):
    import cv2
    from roomscan.stray import StrayCapture
    ivs = intervals(cap)
    d = CACHE / "fullrate" / cap / "rgb"
    d.mkdir(parents=True, exist_ok=True)
    rows = {r for a, b in ivs for r in range(a, b + 1)}
    for r, bgr in StrayCapture.load(ROOT / cap).iter_rgb(rows):
        cv2.imwrite(str(d / f"{r:06d}.jpg"), cv2.resize(bgr, (960, 720), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    (CACHE / "fullrate" / cap / "intervals.json").write_text(json.dumps(ivs))
    print(cap, len(ivs), "intervals,", len(rows), "frames")


def match(cap: str):
    import time
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image
    from common import wait_for_memory
    print(wait_for_memory())
    base = CACHE / "fullrate" / cap
    ivs = json.loads((base / "intervals.json").read_text())
    dev = torch.device("mps")
    ex = ALIKED(max_num_keypoints=2048, detection_threshold=0.01).eval().to(dev)
    m = LightGlue(features="aliked").eval().to(dev)
    out, t0, n = {}, time.time(), 0
    with torch.no_grad():
        for a, b in ivs:
            prev = None
            for r in range(a, b + 1):
                f = ex.extract(load_image(base / "rgb" / f"{r:06d}.jpg").to(dev), resize=None)
                if prev is not None:
                    o = m({"image0": prev[1], "image1": f})
                    m0 = o["matches0"][0].cpu().numpy()
                    v = np.where(m0 > -1)[0]
                    k0 = prev[1]["keypoints"][0].cpu().numpy()[v]
                    k1 = f["keypoints"][0].cpu().numpy()[m0[v]]
                    out[f"{prev[0]}_{r}"] = np.concatenate([k0, k1], 1).astype(np.float32)
                    n += 1
                prev = (r, f)
            torch.mps.empty_cache()
    np.savez_compressed(base / "matches.npz", **out)
    (base / "timing.json").write_text(json.dumps({"pairs": n, "sec": time.time() - t0}))
    print(cap, n, "pairs", round(time.time() - t0, 1), "s")


def rel_rotation(k0, k1, K):
    """Rotation of camera 1 w.r.t. camera 0 (R_1<-0) from the essential matrix; None if too few matches."""
    import cv2
    if len(k0) < 15:
        return None
    E, inl = cv2.findEssentialMat(k0, k1, K, cv2.RANSAC, 0.999, 1.0)
    if E is None or E.shape != (3, 3):
        return None
    _, R, _, _ = cv2.recoverPose(E, k0, k1, K, mask=inl)
    return R


def evaluate(cap: str):
    from roomscan.stray import StrayCapture
    base = CACHE / "fullrate" / cap
    ivs = json.loads((base / "intervals.json").read_text())
    z = np.load(base / "matches.npz")
    sc = StrayCapture.load(ROOT / cap)
    Ts, ts = sc.T_wc(), sc.timestamps
    K = np.median(np.array([f["K"] for f in load_frames(cap)]), axis=0)
    res = []
    for a, b in ivs:
        n_m, R_chain, broken = [], np.eye(3), 0
        for r in range(a, b):
            kk = z[f"{r}_{r + 1}"]
            n_m.append(len(kk))
            R = rel_rotation(kk[:, :2].astype(np.float64), kk[:, 2:].astype(np.float64), K)
            if R is None:
                broken += 1
                R = np.eye(3)
            R_chain = R @ R_chain
        R_gt = Ts[b, :3, :3].T @ Ts[a, :3, :3]
        n_m = np.array(n_m)
        res.append({"rows": [a, b], "t_s": round(float(ts[a] - ts[0]), 1), "dur_s": round(float(ts[b] - ts[a]), 2),
                    "gt_rotation_deg": round(rot_deg(np.eye(3), R_gt), 1),
                    "links": int(len(n_m)), "links_lt20_matches": int((n_m < 20).sum()), "links_no_pose": broken,
                    "min_matches": int(n_m.min()), "median_matches": float(np.median(n_m)),
                    "chained_rot_err_deg": round(rot_deg(R_chain, R_gt), 2) if broken == 0 else None})
    summ = {"intervals": res, "n_intervals": len(res),
            "n_bridged": sum(1 for r in res if r["links_no_pose"] == 0),
            "chained_rot_err_deg_median": float(np.median([r["chained_rot_err_deg"] for r in res
                                                            if r["chained_rot_err_deg"] is not None] or [np.nan]))}
    (OUT / f"{cap}_fullrate.json").write_text(json.dumps(summ, indent=1))
    for r in res:
        print(r)
    print({k: v for k, v in summ.items() if k != "intervals"})


if __name__ == "__main__":
    cmd, cap = sys.argv[1], sys.argv[2]
    {"dump": dump, "match": match, "eval": evaluate}[cmd](cap)

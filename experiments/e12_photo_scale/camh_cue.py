"""Camera-height cue v2: LOWEST well-supported horizontal plane (v1 took the largest, which is
often a table/bed top -> heights 37 % too small).

Uses cached MoGe depth (cache/e12, 256x192 landscape) rotated upright; 'down' = upright image +y
(no gravity available at the photo tier; protocol keeps the phone roughly level).
Reports, per photo: measured camera height in MoGe metric units vs true (LiDAR floor), and the
scale error of the resulting estimator s = H_PRIOR / h_pred, with H_PRIOR = population prior.
"""
import json
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"; CACHE = ROOT / "cache/e12"
F = json.loads((OUT / "moge_per_frame.json").read_text())
meta = {m["row"]: m for m in json.loads((ROOT / "cache/frames/single_scan_with_ceiling/frames.json").read_text())["frames"]}
rng = np.random.default_rng(0)

def lowest_floor(P, tol=0.02, min_frac=0.03, tilt_deg=30):
    cands = []
    for _ in range(600):
        s = P[rng.choice(len(P), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn; n = n if n[1] > 0 else -n
        if n[1] < np.cos(np.radians(tilt_deg)): continue
        d = -n @ s[0]; inl = np.abs(P @ n + d) < tol
        if inl.mean() >= min_frac: cands.append((abs(d), inl.mean(), n))
    if not cands: return None
    dmax = max(c[0] for c in cands)
    low = [c for c in cands if c[0] > dmax - 0.05]          # planes within 5 cm of the lowest
    return float(np.median([c[0] for c in low]))

rows = []
for r, v in F.items():
    r = int(r); d = np.load(CACHE / f"{r:06d}.npy").astype(np.float32)
    up = cv2.rotate(d, cv2.ROTATE_90_CLOCKWISE)                           # 192 wide x 256 high
    K = np.array(meta[r]["K"]); sx = 256 / 960
    fx, fy, cx, cy = K[1, 1] * sx, K[0, 0] * sx, (720 - 1 - K[1, 2]) * sx, K[0, 2] * sx   # landscape->upright (CW)
    H, W = up.shape; vv, uu = np.mgrid[0:H, 0:W]
    m = np.isfinite(up) & (vv > H * 0.5)
    z = up[m]; P = np.stack([(uu[m] - cx) * z / fx, (vv[m] - cy) * z / fy, z], 1)
    h = lowest_floor(P) if len(P) > 300 else None
    rows.append({"row": r, "room": v["room"], "h_pred": h, "h_true": v["cam_h_true"], "mono_err": v["scale_err"]})
(OUT / "camh_v2_per_frame.json").write_text(json.dumps(rows))
ok = [x for x in rows if x["h_pred"]]
hp = np.array([x["h_pred"] for x in ok]); ht = np.array([x["h_true"] for x in ok]); me = np.array([x["mono_err"] for x in ok])
meas_err = hp / ht - 1
# the MoGe height error should equal the MoGe scale error if the floor plane is right
res = {"floor_found_frac": round(len(ok) / len(rows), 3),
       "h_pred_vs_true_err_pct": {"median_signed": round(100 * float(np.median(meas_err)), 2), "median_abs": round(100 * float(np.median(np.abs(meas_err))), 2)},
       "h_err_minus_mono_err_pct_median_abs": round(100 * float(np.median(np.abs(meas_err - me))), 2)}
# estimator: geometry rescaled so camera height = prior. Size error = (H_PRIOR/h_pred)*(1+mono_err) - 1
for name, H in [("prior_1.45", 1.45), ("prior_1.55_oracle_operator_mean", float(ht.mean()))]:
    e = (H / hp) * (1 + me) - 1
    res[f"camh_estimator_{name}"] = {"median_signed_pct": round(100 * float(np.median(e)), 2), "median_abs_pct": round(100 * float(np.median(np.abs(e))), 2),
                                     "p90_abs_pct": round(100 * float(np.percentile(np.abs(e), 90)), 2), "within_8pct": round(float((np.abs(e) <= 0.08).mean()), 3)}
# per-room K=4 fusion of mono and camh (log-space average with equal weights)
by_room = {}
for x in ok: by_room.setdefault(x["room"], []).append(x)
fus = []; mon = []
for room, xs in by_room.items():
    for _ in range(200):
        sub = [xs[i] for i in rng.choice(len(xs), min(4, len(xs)), replace=False)]
        m_ = np.median([s["mono_err"] for s in sub]); c_ = (1.45 / np.median([s["h_pred"] for s in sub])) * (1 + m_) - 1
        fus.append(np.exp(0.5 * (np.log1p(m_) + np.log1p(c_))) - 1); mon.append(m_)
fus, mon = np.array(fus), np.array(mon)
res["K4_room_mono"] = {"median_abs_pct": round(100 * float(np.median(np.abs(mon))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.abs(mon), 90)), 2), "within_8pct": round(float((np.abs(mon) <= .08).mean()), 3)}
res["K4_room_fused_mono_camh"] = {"median_abs_pct": round(100 * float(np.median(np.abs(fus))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.abs(fus), 90)), 2), "within_8pct": round(float((np.abs(fus) <= .08).mean()), 3)}
(OUT / "camh_v2.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

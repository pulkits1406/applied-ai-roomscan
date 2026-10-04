"""Photo-tier scale: does aggregating K photos, or a camera-height cue, reach the ±8 % gate?

Inputs: out/moge_per_frame.json (MoGe-2, FOV given). Pseudo-GT: LiDAR/ARKit.
Estimators for one room given K photos:
  mono        : scale error = median over photos of per-photo scale error (MoGe metric as-is)
  camh        : scale from camera height: s = H_PRIOR / median(cam_h_pred) (photos w/o floor skipped)
  fused       : inverse-variance combination of mono and camh in log space
For each, the systematic part (bias) and spread across random K-subsets per room are reported,
plus leave-one-room-out bias correction (bias estimated on the other rooms) to show whether a
dev-set calibration transfers between rooms.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"
F = json.loads((OUT / "moge_per_frame.json").read_text())
rooms = {}
for r, v in F.items(): rooms.setdefault(v["room"], []).append(v)
rng = np.random.default_rng(0)

allv = list(F.values())
ch_true = np.array([v["cam_h_true"] for v in allv]); ch_pred = np.array([v["cam_h_pred"] if v["cam_h_pred"] else np.nan for v in allv])
se = np.array([v["scale_err"] for v in allv])
summary = {"n_frames": len(allv),
           "per_photo_mono_err_pct": {"median_signed": round(100 * float(np.median(se)), 2), "median_abs": round(100 * float(np.median(np.abs(se))), 2), "p90_abs": round(100 * float(np.percentile(np.abs(se), 90)), 2)},
           "true_camera_height_m": {"mean": round(float(ch_true.mean()), 3), "std": round(float(ch_true.std()), 3), "p5_p95": np.percentile(ch_true, [5, 95]).round(3).tolist()},
           "floor_found_frac": round(float(np.isfinite(ch_pred).mean()), 3)}
ok = np.isfinite(ch_pred)
camh_err = ch_pred[ok] / ch_true[ok] - 1
summary["per_photo_camh_measure_err_pct"] = {"median_signed": round(100 * float(np.median(camh_err)), 2), "median_abs": round(100 * float(np.median(np.abs(camh_err))), 2),
                                            "corr_with_mono_err": round(float(np.corrcoef(camh_err, se[ok])[0, 1]), 3)}
H_PRIOR = 1.45   # protocol "hold the phone at chest height"; this operator's mean is reported above

def room_estimates(vs, K, bias_mono=0.0):
    sub = [vs[i] for i in rng.choice(len(vs), K, replace=False)]
    mono = float(np.median([v["scale_err"] for v in sub])) - bias_mono   # signed error of predicted metric size
    h = [v["cam_h_pred"] for v in sub if v["cam_h_pred"]]
    # camh estimator: rescale MoGe geometry so camera height = prior; error of resulting size:
    camh = (H_PRIOR / np.median(h)) * (1 + float(np.median([v["scale_err"] for v in sub]))) - 1 if h else np.nan
    return mono, camh

res = {"summary": summary, "by_K": {}, "by_room": {}}
for K in [1, 2, 3, 4, 6, 8]:
    rows = []
    for room, vs in rooms.items():
        for _ in range(200): rows.append((room, *room_estimates(vs, K)))
    m = np.array([x[1] for x in rows]); c = np.array([x[2] for x in rows])
    res["by_K"][K] = {"mono": {"median_signed_pct": round(100 * float(np.median(m)), 2), "median_abs_pct": round(100 * float(np.median(np.abs(m))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.abs(m), 90)), 2), "within_8pct": round(float((np.abs(m) <= 0.08).mean()), 3)},
                      "camh": {"median_signed_pct": round(100 * float(np.nanmedian(c)), 2), "median_abs_pct": round(100 * float(np.nanmedian(np.abs(c))), 2), "p90_abs_pct": round(100 * float(np.nanpercentile(np.abs(c), 90)), 2), "within_8pct": round(float((np.abs(c[np.isfinite(c)]) <= 0.08).mean()), 3)}}
for room, vs in rooms.items():
    s = np.array([v["scale_err"] for v in vs])
    res["by_room"][room] = {"n": len(vs), "mono_median_signed_pct": round(100 * float(np.median(s)), 2), "camh_true_mean_m": round(float(np.mean([v["cam_h_true"] for v in vs])), 3)}
# leave-one-room-out bias correction for mono
loro = []
for room in rooms:
    bias = np.median([v["scale_err"] for rr, vs in rooms.items() if rr != room for v in vs])
    for _ in range(200):
        loro.append(room_estimates(rooms[room], 4, bias_mono=bias)[0])
loro = np.array(loro)
res["mono_K4_leave_one_room_out_bias_corrected"] = {"median_abs_pct": round(100 * float(np.median(np.abs(loro))), 2), "p90_abs_pct": round(100 * float(np.percentile(np.abs(loro), 90)), 2), "within_8pct": round(float((np.abs(loro) <= 0.08).mean()), 3)}
(OUT / "analysis.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

"""Part A tables and Part B analyses from out/runs.jsonl (+ out/crop_probe.jsonl). No GPU.

  cache/e13/.venv/bin/python experiments/e16_photo_scale_generalization/summarize.py
Writes out/summary.json, out/per_room_table.md, out/per_room_err_vs_K.png.
All errors are vs pseudo-GT (LiDAR conf==2 depth, ARKit centres); |err| < ~3 % is below the GT floor.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
runs = [json.loads(l) for l in open(OUT / "runs.jsonl")]
by_key = {(r["room"], r["K"], r["subset"]): r for r in runs}  # last record per (room, K, subset)
runs = list(by_key.values())
METHODS = {"MA-raw": "scale_err_pct", "MA-fcorr": "scale_err_pct_focalcorr", "MoGe-2": "moge_err_pct"}
CAT = 25.0


def stats(e):
    e = np.asarray(e, float)
    return {"n": len(e), "abs_med": round(float(np.median(np.abs(e))), 1), "abs_p90": round(float(np.percentile(np.abs(e), 90)), 1),
            "bias_med": round(float(np.median(e)), 1), "within8": round(float(100 * np.mean(np.abs(e) <= 8)), 0),
            "n_cat": int(np.sum(np.abs(e) > CAT))}


rooms = sorted({r["room"] for r in runs}, key=int)
Ks = sorted({r["K"] for r in runs})
S = {"per_room": {}, "pooled": {}, "notes": {"cat_threshold_pct": CAT, "gt_floor": "LiDAR vs ARKit scale disagree ~3 %"}}
lines = ["| room | K | n | MA-raw med/p90/bias/≤8% | MA-fcorr med/p90/bias/≤8% | MoGe-2 med/p90/bias/≤8% | cat (raw/fc/MoGe) | s/run |",
         "|---|---|---|---|---|---|---|---|"]
for room in rooms:
    S["per_room"][room] = {}
    for K in Ks:
        g = [r for r in runs if r["room"] == room and r["K"] == K]
        if not g:
            continue
        rec = {m: stats([r[k] for r in g]) for m, k in METHODS.items()}
        rec["sec_median"] = round(float(np.median([r["sec"] for r in g])), 1)
        S["per_room"][room][K] = rec
        f = lambda s: f"{s['abs_med']}/{s['abs_p90']}/{s['bias_med']:+}/{s['within8']:.0f}"
        lines.append(f"| {room} | {K} | {len(g)} | {f(rec['MA-raw'])} | {f(rec['MA-fcorr'])} | {f(rec['MoGe-2'])} | "
                     f"{rec['MA-raw']['n_cat']}/{rec['MA-fcorr']['n_cat']}/{rec['MoGe-2']['n_cat']} | {rec['sec_median']} |")
for K in Ks:
    g = [r for r in runs if r["K"] == K]
    S["pooled"][K] = {m: stats([r[k] for r in g]) for m, k in METHODS.items()}
S["pooled_all"] = {m: stats([r[k] for r in runs]) for m, k in METHODS.items()}
S["n_catastrophic_runs"] = {m: int(sum(abs(r[k]) > CAT for r in runs)) for m, k in METHODS.items()}
S["catastrophic_runs_fcorr_or_moge"] = [{"room": r["room"], "K": r["K"], "subset": r["subset"], "rows": r["rows"],
                                         **{m: round(r[k], 1) for m, k in METHODS.items()}}
                                        for r in runs if abs(r["scale_err_pct_focalcorr"]) > CAT or abs(r["moge_err_pct"]) > CAT]

# ---- Part B(a): focal ratio f_pred/f_true per view.
fr_all, fr_room, fr_K = [], defaultdict(list), defaultdict(list)
for r in runs:
    for x in r["per_image_focal_ratio"]:
        fr_all.append(x); fr_room[r["room"]].append(x); fr_K[r["K"]].append(x)
d = lambda v: {"n": len(v), "mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3),
               "p5": round(float(np.percentile(v, 5)), 3), "p95": round(float(np.percentile(v, 95)), 3)}
S["B_a_focal_ratio"] = {"all_views": d(fr_all), "per_room": {k: d(v) for k, v in sorted(fr_room.items())},
                        "per_K": {k: d(v) for k, v in sorted(fr_K.items())},
                        "fy_over_fx_pred_vs_true_median": round(float(np.median(
                            [y / x for r in runs for x, y in zip(r["per_image_focal_ratio"], r["per_image_focal_ratio_y"])])), 4)}

# ---- Part B(b): depth scale vs focal vs pose scale.
rb = [r for r in runs if r.get("baseline_ratio_pred_over_gt")]
rd = np.array([r["ratio_pred_over_lidar"] for r in rb]); rf = np.array([np.median(r["per_image_focal_ratio"]) for r in rb])
bb = np.array([r["baseline_ratio_pred_over_gt"] for r in rb])
# Per-view: does each view's depth ratio track its own focal ratio within a run? (deviations from run median)
dv, fv = [], []
for r in runs:
    if r["K"] < 2:
        continue
    pr = np.array([x if x else np.nan for x in r["per_image_ratio"]]); pf = np.array(r["per_image_focal_ratio"])
    ok = np.isfinite(pr)
    dv += list(np.log(pr[ok]) - np.median(np.log(pr[ok]))); fv += list(np.log(pf[ok]) - np.median(np.log(pf[ok])))
q = lambda v: {"median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3), "p90": round(float(np.percentile(v, 90)), 3)}
S["B_b_scale_decomposition"] = {
    "n_runs": len(rb),
    "depth_ratio_pred_over_lidar": q(rd), "focal_ratio": q(rf), "baseline_ratio_pred_over_arkit": q(bb),
    "depth_over_focal (=fcorr ratio)": q(rd / rf), "baseline_over_focal": q(bb / rf), "baseline_over_depth": q(bb / rd),
    "corr_log_depth_vs_log_focal_across_runs": round(float(np.corrcoef(np.log(rd), np.log(rf))[0, 1]), 3),
    "corr_log_baseline_vs_log_depth_across_runs": round(float(np.corrcoef(np.log(bb), np.log(rd))[0, 1]), 3),
    "within_run_per_view_corr_log_depth_vs_log_focal": round(float(np.corrcoef(dv, fv)[0, 1]), 3),
    "within_run_per_view_slope_log_depth_on_log_focal": round(float(np.polyfit(fv, dv, 1)[0]), 3),
    "n_views_within_run": len(dv)}
S["B_b_scale_decomposition"]["per_room_K>=4"] = {
    room: {"depth_ratio": round(float(np.median([r["ratio_pred_over_lidar"] for r in g])), 3),
           "focal_ratio": round(float(np.median([np.median(r["per_image_focal_ratio"]) for r in g])), 3),
           "baseline_ratio": round(float(np.median([r["baseline_ratio_pred_over_gt"] for r in g])), 3),
           "baseline_ratio_p10_p90": [round(float(np.percentile([r["baseline_ratio_pred_over_gt"] for r in g], q)), 3) for q in (10, 90)]}
    for room in rooms for g in [[r for r in rb if r["room"] == room and r["K"] >= 4]]}
s3 = [r for r in runs if "sim3_ratio_pred_over_gt" in r]
if s3:
    S["B_b_scale_decomposition"]["sim3_ratio_pred_over_arkit"] = {**q([r["sim3_ratio_pred_over_gt"] for r in s3]), "n": len(s3),
                                                                  "over_depth": q([r["sim3_ratio_pred_over_gt"] / r["ratio_pred_over_lidar"] for r in s3])}

# ---- Part B(a'): which focal to correct with? Recompute from per-view caches (fx, fy, sqrt(fx fy), run-median fx).
CACHE = Path(__file__).resolve().parents[2] / "cache/e16/runs"
if CACHE.exists():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from e16lib import META, MODEL_SCALE, samples
    var = defaultdict(list)
    for r in runs:
        z = np.load(CACHE / f"r{r['room']}_K{r['K']}_s{r['subset']}.npz")
        rows = [int(x) for x in z["rows"]]
        P, L = zip(*[samples(z["depth_land"][i], row) for i, row in enumerate(rows)])
        fxr = z["K_pred"][:, 0, 0] / np.array([META[row]["K"][1][1] * MODEL_SCALE for row in rows])
        fyr = z["K_pred"][:, 1, 1] / np.array([META[row]["K"][0][0] * MODEL_SCALE for row in rows])
        La = np.concatenate(L)
        for name, f in [("fx", fxr), ("fy", fyr), ("sqrt_fxfy", np.sqrt(fxr * fyr)), ("run_median_fx", np.full_like(fxr, np.median(fxr)))]:
            var[(name, r["room"])].append(100 * (np.median(np.concatenate([p / fi for p, fi in zip(P, f)]) / La) - 1))
    S["B_a_fcorr_variants_abs_med_by_room"] = {name: {room: round(float(np.median(np.abs(var[(name, room)]))), 1) for room in rooms}
                                               for name in ["fx", "fy", "sqrt_fxfy", "run_median_fx"]}
    S["B_a_fcorr_variants_bias_by_room"] = {name: {room: round(float(np.median(var[(name, room)])), 1) for room in rooms}
                                            for name in ["fx", "fy", "sqrt_fxfy", "run_median_fx"]}

# ---- Part B(c): held-out constant correction (train rooms 1,3,4 -> test 6,7,8).
train, test = {"1", "3", "4"}, {"6", "7", "8"}
hc = {}
for Kset, name in [(Ks, "all_K")] + [([k], f"K{k}") for k in Ks]:
    tr = [r for r in runs if r["room"] in train and r["K"] in Kset]
    te = [r for r in runs if r["room"] in test and r["K"] in Kset]
    if not tr or not te:
        continue
    c = float(np.median([1 / r["ratio_pred_over_lidar"] for r in tr]))
    e_const = [100 * (r["ratio_pred_over_lidar"] * c - 1) for r in te]
    hc[name] = {"c_lidar_over_pred": round(c, 4), "const": stats(e_const),
                "fcorr_exif": stats([r["scale_err_pct_focalcorr"] for r in te]), "moge": stats([r["moge_err_pct"] for r in te]),
                "const_per_test_room_bias": {room: round(float(np.median([100 * (r["ratio_pred_over_lidar"] * c - 1) for r in te if r["room"] == room])), 1)
                                             for room in sorted(test) if any(r["room"] == room for r in te)}}
S["B_c_heldout_constant"] = hc

# ---- Part B(d): crop probe.
cp = OUT / "crop_probe.jsonl"
if cp.exists():
    cr = [json.loads(l) for l in open(cp)]
    S["B_d_crop_probe"] = {str(c): {"n": len(g), "fov_true": round(float(np.median([r["fov_true_deg"] for r in g])), 1),
                                    "fov_pred_median": round(float(np.median([r["fov_pred_deg"] for r in g])), 1),
                                    "fov_pred_range": [round(min(r["fov_pred_deg"] for r in g), 1), round(max(r["fov_pred_deg"] for r in g), 1)],
                                    "focal_ratio_median": round(float(np.median([r["focal_ratio_median"] for r in g])), 3),
                                    "raw_err_median": round(float(np.median([r["scale_err_pct"] for r in g])), 1),
                                    "fcorr_err_median": round(float(np.median([r["scale_err_pct_focalcorr"] for r in g])), 1),
                                    "fcorr_abs_err_max": round(float(max(abs(r["scale_err_pct_focalcorr"]) for r in g)), 1)}
                           for c in sorted({r["crop"] for r in cr}, reverse=True) for g in [[r for r in cr if r["crop"] == c]]}

(OUT / "summary.json").write_text(json.dumps(S, indent=1))
(OUT / "per_room_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
print(json.dumps({k: S[k] for k in ["pooled_all", "B_c_heldout_constant"]}))
if "B_d_crop_probe" in S:
    print(json.dumps(S["B_d_crop_probe"], indent=1))
print("catastrophic:", S["n_catastrophic_runs"])
for k in ["B_a_fcorr_variants_abs_med_by_room", "B_a_fcorr_variants_bias_by_room"]:
    print(k, json.dumps(S.get(k)))
print(json.dumps(S["B_b_scale_decomposition"]["per_room_K>=4"]))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for a, (m, key) in zip(ax, METHODS.items()):
    for i, room in enumerate(rooms):
        kk = [K for K in Ks if K in S["per_room"][room]]
        a.plot(kk, [S["per_room"][room][K][m]["abs_med"] for K in kk], "o-", c=plt.cm.tab10(i), label=f"room {room}")
    a.axhline(8, c="k", lw=0.8, ls=":"); a.axhline(3, c="gray", lw=0.8, ls="--"); a.set_title(m); a.set_xlabel("K photos")
ax[0].set_ylabel("median |scale err| % (pseudo-GT)"); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "per_room_err_vs_K.png", dpi=100)

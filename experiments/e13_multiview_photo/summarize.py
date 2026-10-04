"""Summarise out/runs.jsonl into out/summary.json and plots (all numbers vs LiDAR/ARKit pseudo-GT)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
MOGE_MEDIAN, MOGE_P90 = 6.4, 21.0  # e12 MoGe-2 single-image, FOV given (reference)

runs = [json.loads(l) for l in open(OUT / "runs.jsonl")]
runs = [r for r in runs if r.get("amp", "bf16") == "bf16"]
groups = defaultdict(list)
for r in runs:
    groups[(r["variant"], r["K"])].append(r)


def pct(x, q):
    return round(float(np.percentile(x, q)), 2)


summary = {"note": "errors are % of predicted vs pseudo-GT (LiDAR depth conf==2, ARKit centres); "
                   "scale_err = 100*(median(pred/lidar)-1), negative = model under-estimates size", "per_variant": {}}
for variant in ["images", "images+K"]:
    sv = {}
    for K in sorted({k for v, k in groups if v == variant}):
        g = groups[(variant, K)]
        e = np.array([r["scale_err_pct"] for r in g])
        rec = {"n_runs": len(g), "abs_err_median": pct(np.abs(e), 50), "abs_err_p90": pct(np.abs(e), 90),
               "signed_err_mean": round(float(e.mean()), 2), "signed_err_median": pct(e, 50),
               "lidar_over_pred_median": round(float(np.median([1 / r["ratio_pred_over_lidar"] for r in g])), 4),
               "frac_within_8pct": round(float(np.mean(np.abs(e) <= 8)), 3),
               "absrel_aligned_median": round(float(np.median([r["absrel_aligned"] for r in g])), 4),
               "sec_median": round(float(np.median([r["sec"] for r in g])), 2),
               "focal_err_pct_median": round(float(np.median([r["focal_err_pct"] for r in g])), 2)}
        # Spread of per-image errors inside one run (does one reconstruction share one scale?)
        within = [np.ptp([x for x in r["per_image_ratio"] if x]) * 100 for r in g if K > 1]
        if within:
            rec["within_run_per_image_range_pct_median"] = round(float(np.median(within)), 2)
        fc = np.array([r["scale_err_pct_focalcorr"] for r in g])
        rec["focalcorr"] = {"abs_err_median": pct(np.abs(fc), 50), "abs_err_p90": pct(np.abs(fc), 90),
                            "signed_err_mean": round(float(fc.mean()), 2), "frac_within_8pct": round(float(np.mean(np.abs(fc) <= 8)), 3)}
        b = [r["baseline_ratio_pred_over_gt"] for r in g if r.get("baseline_ratio_pred_over_gt")]
        if b:
            rec["baseline_ratio_pred_over_gt_median"] = round(float(np.median(b)), 3)
        s3 = [100 * (r["sim3_ratio_pred_over_gt"] - 1) for r in g if "sim3_ratio_pred_over_gt" in r]
        if s3:
            d = [abs(100 * (r["sim3_ratio_pred_over_gt"] - 1) - r["scale_err_pct"]) for r in g if "sim3_ratio_pred_over_gt" in r]
            rec["sim3"] = {"n": len(s3), "signed_err_median": pct(s3, 50), "abs_err_median": pct(np.abs(s3), 50),
                           "abs_err_p90": pct(np.abs(s3), 90), "abs_diff_vs_depth_scale_median_pp": pct(d, 50),
                           "rmse_m_median": round(float(np.median([r["sim3_rmse_m"] for r in g if "sim3_rmse_m" in r])), 3)}
        rooms = defaultdict(list)
        for r in g:
            rooms[r["room"]].append(r["scale_err_pct"])
        rec["per_room_signed_err_mean"] = {k: round(float(np.mean(v)), 2) for k, v in sorted(rooms.items())}
        rec["per_room_abs_err_median"] = {k: round(float(np.median(np.abs(v))), 2) for k, v in sorted(rooms.items())}
        sv[str(K)] = rec
    summary["per_variant"][variant] = sv
# Paired comparison on identical subsets.
pairs = defaultdict(dict)
for r in runs:
    pairs[(r["room"], tuple(r["rows"]))][r["variant"]] = r["scale_err_pct"]
d = [abs(v["images+K"]) - abs(v["images"]) for v in pairs.values() if len(v) == 2]
summary["paired_abs_err_diff_withK_minus_without_pp"] = {"n": len(d), "median": pct(d, 50), "frac_withK_better": round(float(np.mean(np.array(d) < 0)), 3)}
summary["coverage"] = {"rooms_run": sorted({r["room"] for r in runs}), "rooms_missing": ["3", "4", "6", "7", "8"],
                       "note": "sweep stopped for machine memory pressure; room 1 only, K=8 has 2 (images) / 1 (images+K) runs; stitching probe not run"}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
for variant, c in [("images", "#d1495b"), ("images+K", "#00798c")]:
    sv = summary["per_variant"][variant]; Ks = [int(k) for k in sv]
    ax[0].plot(Ks, [sv[str(k)]["abs_err_median"] for k in Ks], "o-", c=c, label=f"{variant} median")
    ax[0].plot(Ks, [sv[str(k)]["abs_err_p90"] for k in Ks], "s--", c=c, alpha=0.6, label=f"{variant} p90")
    rooms = sorted({r for k in sv for r in sv[k]["per_room_signed_err_mean"]})
    for i, room in enumerate(rooms):
        ax[1].plot(Ks, [sv[str(k)]["per_room_signed_err_mean"].get(room, np.nan) for k in Ks], "o-" if variant == "images+K" else "x:",
                   c=plt.cm.tab10(i), label=f"room {room}" if variant == "images+K" else None)
ax[0].axhline(MOGE_MEDIAN, c="gray", lw=1); ax[0].axhline(MOGE_P90, c="gray", lw=1, ls="--")
ax[0].text(1, MOGE_MEDIAN + 0.4, "MoGe-2 1-img median", fontsize=7, c="gray")
ax[0].axhline(8, c="k", lw=0.8, ls=":"); ax[0].set_xlabel("K photos"); ax[0].set_ylabel("|scale err| % vs LiDAR (pseudo-GT)")
ax[0].legend(fontsize=7); ax[0].set_title("MapAnything metric scale error")
ax[1].axhline(0, c="k", lw=0.8); ax[1].set_xlabel("K photos"); ax[1].set_ylabel("mean signed scale err %")
ax[1].set_title("per-room bias (solid: +K, dotted: images only)", fontsize=9); ax[1].legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(OUT / "scale_err_vs_K.png", dpi=110)
print(json.dumps(summary, indent=1))

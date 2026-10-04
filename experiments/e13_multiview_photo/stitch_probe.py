"""Stitching probe: one MapAnything run on 3 photos of room A + 3 photos of an adjacent room B.

Adjacency: label regions (e10 v2b labels referenced by photo_sets.json) touch after a
6-cell (0.30 m) dilation, which bridges the unlabelled wall band between rooms.
Per run, against ARKit pseudo-GT:
- Sim(3)-align all 6 predicted centres to ARKit; per-camera position error (m).
- Inter-room offset error: |(mean_B - mean_A)_aligned - (mean_B - mean_A)_ARKit| (m).
- Cross-room relative rotation error: angle of (Ra^T Rb)_pred^-1 (Ra^T Rb)_ARKit over A x B pairs
  (world-frame free, so it isolates the rooms' relative orientation).
- Depth scale error per room (pred / LiDAR), as in run_subsets.py.
Writes out/stitch_runs.jsonl and out/stitch_summary.json.
"""
from __future__ import annotations

import json

import numpy as np
from scipy import ndimage as ndi

from common import OUT, ROOT, R_wc_upright, centre, load_model, pred_vs_lidar, run_model, umeyama
from run_subsets import draw_subsets

N_PER_PAIR = 4
sets_doc = json.loads((ROOT / "experiments/e12_photo_scale/out/photo_sets.json").read_text())
sets = sets_doc["rooms"]
lab = np.load(ROOT / sets_doc["label_file"])
rooms = sorted(int(r) for r in sets)
pairs = [(a, b) for a in rooms for b in rooms
         if b > a and (ndi.binary_dilation(lab == a, iterations=6) & (lab == b)).any()]
print("adjacent pairs", pairs)


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


model = load_model("mps")
recs = []
with open(OUT / "stitch_runs.jsonl", "w") as f:
    for a, b in pairs:
        A = draw_subsets(sets[str(a)], 3, N_PER_PAIR, seed=7000 + 10 * a + b)
        B = draw_subsets(sets[str(b)], 3, N_PER_PAIR, seed=8000 + 10 * a + b)
        for si, (ra, rb) in enumerate(zip(A, B)):
            rows = ra + rb
            for variant in ["images", "images+K"]:
                res = run_model(model, rows, with_K=variant == "images+K")
                cp, cg = np.array(res["centre"]), np.array([centre(r) for r in rows])
                s, R, t = umeyama(cp, cg)
                ca = s * cp @ R.T + t
                err = np.linalg.norm(ca - cg, axis=1)
                off_err = float(np.linalg.norm((ca[3:].mean(0) - ca[:3].mean(0)) - (cg[3:].mean(0) - cg[:3].mean(0))))
                gt_off = float(np.linalg.norm(cg[3:].mean(0) - cg[:3].mean(0)))
                Rp, Rg = res["R"], [R_wc_upright(r) for r in rows]
                rel = [rot_err_deg(Rp[i].T @ Rp[j], Rg[i].T @ Rg[j]) for i in range(3) for j in range(3, 6)]
                scale = {}
                for name, idx in [("A", range(3)), ("B", range(3, 6))]:
                    P, L = zip(*[pred_vs_lidar(res["depth"][i], res["mask"][i], rows[i]) for i in idx])
                    scale[name] = float(100 * (np.median(np.concatenate(P) / np.concatenate(L)) - 1))
                rec = {"pair": [a, b], "subset": si, "variant": variant, "rows": rows, "sec": round(res["sec"], 2),
                       "sim3_ratio_pred_over_gt": 1 / s, "cam_err_m": err.round(3).tolist(),
                       "cam_err_max_m": float(err.max()), "room_offset_gt_m": gt_off, "room_offset_err_m": off_err,
                       "cross_rel_rot_err_deg_median": float(np.median(rel)), "cross_rel_rot_err_deg_max": float(np.max(rel)),
                       "scale_err_pct_A": scale["A"], "scale_err_pct_B": scale["B"]}
                recs.append(rec); f.write(json.dumps(rec) + "\n"); f.flush()
                print(a, b, si, variant, "cam err", err.round(2), f"offset err {off_err:.2f}/{gt_off:.2f} m",
                      f"rel rot med {rec['cross_rel_rot_err_deg_median']:.1f} deg", f"{res['sec']:.1f}s", flush=True)

summ = {}
for variant in ["images", "images+K"]:
    rr = [r for r in recs if r["variant"] == variant]
    errs = np.concatenate([r["cam_err_m"] for r in rr])
    summ[variant] = {"n_runs": len(rr), "cam_err_median_m": round(float(np.median(errs)), 3),
                     "cam_err_p90_m": round(float(np.percentile(errs, 90)), 3),
                     "room_offset_err_median_m": round(float(np.median([r["room_offset_err_m"] for r in rr])), 3),
                     "cross_rel_rot_err_median_deg": round(float(np.median([r["cross_rel_rot_err_deg_median"] for r in rr])), 2),
                     "runs_all_cams_within_0.5m": int(sum(r["cam_err_max_m"] < 0.5 for r in rr)),
                     "runs_any_cam_over_1m": int(sum(r["cam_err_max_m"] > 1.0 for r in rr)),
                     "per_pair": {f"{a}-{b}": {"cam_err_median_m": round(float(np.median(np.concatenate([r["cam_err_m"] for r in rr if r["pair"] == [a, b]]))), 3),
                                               "room_offset_err_m": [round(r["room_offset_err_m"], 2) for r in rr if r["pair"] == [a, b]]}
                                  for a, b in pairs}}
summ["adjacent_pairs"] = pairs
summ["note"] = "pseudo-GT = ARKit poses / LiDAR depth from Stray Scanner"
(OUT / "stitch_summary.json").write_text(json.dumps(summ, indent=1))
print(json.dumps(summ, indent=1))

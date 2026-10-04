"""MapAnything metric scale on K-photo subsets per room vs LiDAR / ARKit pseudo-GT.

For each room and K, draws seeded subsets with one photo per contiguous time segment of the
room's candidate list (mimics walking around the room), runs MapAnything (a) images only and
(b) images + known intrinsics on the same subsets, and appends one JSON line per run to
out/runs.jsonl. Re-running skips runs already present, so it can be split across invocations.

  cache/e13/.venv/bin/python experiments/e13_multiview_photo/run_subsets.py --rooms 1 3 --ks 1 2 3 4 6 8
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from common import META, OUT, ROOT, centre, load_model, pred_vs_lidar, run_model, spread_ok, umeyama


def draw_subsets(rows: list[int], K: int, n: int, seed: int) -> list[list[int]]:
    rows = sorted(rows)
    rng = np.random.default_rng(seed)
    segs = np.array_split(np.arange(len(rows)), K)
    out, seen = [], set()
    for _ in range(50 * n):
        pick = tuple(rows[int(rng.choice(s))] for s in segs)
        if pick not in seen:
            seen.add(pick); out.append(list(pick))
        if len(out) == n:
            break
    return out


def evaluate(rows: list[int], res: dict) -> dict:
    P, L, per_img = [], [], []
    for i, r in enumerate(rows):
        p, l = pred_vs_lidar(res["depth"][i], res["mask"][i], r)
        P.append(p); L.append(l)
        per_img.append(float(np.median(p / l)) if len(p) > 50 else None)
    P_list = P
    P, L = np.concatenate(P), np.concatenate(L)
    ratio = float(np.median(P / L))  # pred / LiDAR
    s = 1.0 / ratio
    rec = {"rows": rows, "K": len(rows), "sec": round(res["sec"], 3), "n_px": int(len(P)),
           "ratio_pred_over_lidar": ratio, "scale_err_pct": 100 * (ratio - 1),
           "per_image_ratio": per_img,
           "absrel_aligned": float(np.mean(np.abs(s * P - L) / L)),
           "absrel_aligned_median": float(np.median(np.abs(s * P - L) / L))}
    # Predicted focal vs true (model resolution 378x504 = 0.525 x upright 720x960).
    fx_true = np.array([META[r]["K"][1][1] * 0.525 for r in rows])
    fx_pred = np.array([k[0, 0] for k in res["K"]])
    rec["focal_err_pct"] = float(100 * np.median(fx_pred / fx_true - 1))
    rec["per_image_focal_ratio"] = (fx_pred / fx_true).tolist()
    # Pinhole size/depth ambiguity: Z = f * S / s, so a model that mis-estimates f scales depth by
    # the same factor. Post-hoc correction rescales each view's depth by f_true / f_pred.
    Pc = np.concatenate([p * (ft / fp) for p, ft, fp in zip(P_list, fx_true, fx_pred)])
    rec["ratio_focalcorr"] = float(np.median(Pc / L))
    rec["scale_err_pct_focalcorr"] = 100 * (rec["ratio_focalcorr"] - 1)
    cp, cg = np.array(res["centre"]), np.array([centre(r) for r in rows])
    if len(rows) >= 2:
        bp = np.linalg.norm(cp[:, None] - cp[None], axis=-1)[np.triu_indices(len(rows), 1)]
        bg = np.linalg.norm(cg[:, None] - cg[None], axis=-1)[np.triu_indices(len(rows), 1)]
        keep = bg > 0.3
        rec["baseline_ratio_pred_over_gt"] = float(np.median(bp[keep] / bg[keep])) if keep.any() else None
    if len(rows) >= 3 and spread_ok(cg):
        sg, R, t = umeyama(cp, cg)  # sg maps pred -> ARKit metres
        rec["sim3_ratio_pred_over_gt"] = 1.0 / sg
        rec["sim3_rmse_m"] = float(np.sqrt(np.mean(np.sum((cg - (sg * cp @ R.T + t)) ** 2, 1))))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="*", default=None)
    ap.add_argument("--ks", nargs="*", type=int, default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--variants", nargs="*", default=["images", "images+K"])
    ap.add_argument("--device", default="mps")
    ap.add_argument("--amp", default="bf16")
    ap.add_argument("--out", default=str(OUT / "runs.jsonl"))
    a = ap.parse_args()
    sets = json.loads((ROOT / "experiments/e12_photo_scale/out/photo_sets.json").read_text())["rooms"]
    rooms = a.rooms or list(sets)
    OUT.mkdir(parents=True, exist_ok=True)
    done = set()
    try:
        for line in open(a.out):
            d = json.loads(line); done.add((d["room"], d["variant"], tuple(d["rows"]), d.get("amp", "bf16")))
    except FileNotFoundError:
        pass
    model = load_model(a.device)
    with open(a.out, "a") as f:
        for room in rooms:
            for K in a.ks:
                for si, rows in enumerate(draw_subsets(sets[room], K, a.n, seed=1000 * int(room) + K)):
                    for variant in a.variants:
                        if (room, variant, tuple(rows), a.amp) in done:
                            continue
                        res = run_model(model, rows, with_K=variant == "images+K", amp=a.amp)
                        rec = {"room": room, "variant": variant, "subset": si, "amp": a.amp, **evaluate(rows, res)}
                        f.write(json.dumps(rec) + "\n"); f.flush()
                        print(room, K, si, variant, f"err {rec['scale_err_pct']:+.1f}%",
                              f"focal {rec['focal_err_pct']:+.1f}% fcorr {rec['scale_err_pct_focalcorr']:+.1f}%",
                              f"sim3 {rec.get('sim3_ratio_pred_over_gt', float('nan')):.3f}",
                              f"absrel {rec['absrel_aligned']:.3f} {rec['sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()

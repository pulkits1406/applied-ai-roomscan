"""Part B(d): does MapAnything's predicted focal / depth scale track the true FOV?

Centre-crop each upright 720x960 image to fraction c and resize back to 720x960. This narrows the true
FOV deterministically (f_true -> f/c) while the true z-depth of every visible point is unchanged.
Inputs: images + the crop-correct K. Rooms 3, 6, 8; subsets 0-1 of the K=3 sweep; c in {1, 0.75, 0.5}.
Predictions are pasted back into the uncropped frame so the LiDAR comparison uses the same pixels.
Caches cache/e16/crop/*.npz, writes out/crop_probe.jsonl.

  cache/e13/.venv/bin/python experiments/e16_photo_scale_generalization/crop_probe.py
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from e16lib import (META, MODEL_SCALE, ROOT, SETS, infer_views, load_model, samples, to_land, upright_K,
                    upright_rgb, wait_for_memory)
from run_subsets import draw_subsets

OUT = ROOT / "experiments/e16_photo_scale_generalization/out"
CACHE = ROOT / "cache/e16/crop"
W, H = 720, 960


def crop(img, K, c):
    w, h = round(W * c), round(H * c); x0, y0 = (W - w) // 2, (H - h) // 2
    im = cv2.resize(img[y0:y0 + h, x0:x0 + w], (W, H), interpolation=cv2.INTER_LINEAR)
    Kc = K.copy(); Kc[0, 0] /= c; Kc[1, 1] /= c; Kc[0, 2] = (K[0, 2] - x0) / c; Kc[1, 2] = (K[1, 2] - y0) / c
    return im, Kc


def paste(depth_up, mask_up, c):
    """Model-res prediction of the crop -> model-res canvas of the full frame (NaN outside the crop)."""
    hm, wm = depth_up.shape; w, h = round(wm * c), round(hm * c); x0, y0 = (wm - w) // 2, (hm - h) // 2
    d = np.where(mask_up, depth_up, np.nan).astype(np.float32)
    canvas = np.full_like(d, np.nan)
    canvas[y0:y0 + h, x0:x0 + w] = cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST)
    return canvas


model, recs = None, []
for room in ["3", "6", "8"]:
    for si, rows in enumerate(draw_subsets(SETS["rooms"][room], 3, 10, seed=1000 * int(room) + 3)[:2]):
        for c in [1.0, 0.75, 0.5]:
            npz = CACHE / f"r{room}_K3_s{si}_c{c}.npz"
            if not npz.exists():
                wait_for_memory()
                model = model or load_model("mps")
                ims, Ks = zip(*[crop(upright_rgb(r), upright_K(r), c) for r in rows])
                res = infer_views(model, list(ims), list(Ks))
                CACHE.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(npz, rows=rows, c=c, sec=res["sec"], K_pred=np.stack(res["K"]), poses=np.stack(res["poses"]),
                                    depth_land=np.stack([to_land(paste(d, m, c), np.ones_like(m)) for d, m in
                                                         zip(res["depth_up"], res["mask_up"])]).astype(np.float16))
            z = np.load(npz)
            P, L = zip(*[samples(z["depth_land"][i], r) for i, r in enumerate(rows)])
            fx_true = np.array([META[r]["K"][1][1] * MODEL_SCALE / c for r in rows])
            fr = z["K_pred"][:, 0, 0] / fx_true
            Pc = np.concatenate([p / f for p, f in zip(P, fr)]); Pa, La = np.concatenate(P), np.concatenate(L)
            fov = lambda f: float(np.degrees(2 * np.arctan(378 / (2 * f))))
            rec = {"room": room, "subset": si, "rows": rows, "crop": c, "n_px": int(len(Pa)),
                   "fov_true_deg": fov(np.median(fx_true)), "fov_pred_deg": fov(np.median(z["K_pred"][:, 0, 0])),
                   "focal_ratio_median": float(np.median(fr)), "scale_err_pct": float(100 * (np.median(Pa / La) - 1)),
                   "scale_err_pct_focalcorr": float(100 * (np.median(Pc / La) - 1))}
            recs.append(rec)
            print(room, si, c, f"fov true {rec['fov_true_deg']:.1f} pred {rec['fov_pred_deg']:.1f}",
                  f"focal ratio {rec['focal_ratio_median']:.3f} err {rec['scale_err_pct']:+.1f}% fcorr {rec['scale_err_pct_focalcorr']:+.1f}%", flush=True)
with open(OUT / "crop_probe.jsonl", "w") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")

"""Method 2: can two room folders be linked by local features? SIFT + ratio test + RANSAC fundamental.

Uses the same photos as run_stitch.py (union of its 3 subsets per room) and the doorway photos.
Signals per doorway pair (A, B):
  door_vs_B / door_vs_A : verified inliers between a doorway photo and each room photo
  A_vs_B                : best inliers over all A x B photo pairs (no doorway photo)
  control               : doorway photo vs photos of rooms outside the pair (false-link floor)
Images only (upright, 480x640). Writes out/sift_links.json.

  uv run python experiments/e17_photo_stitch/sift_link.py
"""
from __future__ import annotations

import json
from itertools import product

import cv2
import numpy as np

from run_stitch import DOORS, HERE, N_PER_ROOM, SETS, draw_subsets, plan, upright_rgb

sift = cv2.SIFT_create(nfeatures=4000)
_feat = {}


def feats(row):
    if row not in _feat:
        g = cv2.cvtColor(cv2.resize(upright_rgb(row), (480, 640), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        _feat[row] = sift.detectAndCompute(g, None)
    return _feat[row]


_bf = cv2.BFMatcher(cv2.NORM_L2)


def inliers(r1, r2) -> int:
    (k1, d1), (k2, d2) = feats(r1), feats(r2)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0
    good = [m for m, n in (p for p in _bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return 0
    p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
    try:
        F, mask = cv2.findFundamentalMat(p1, p2, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:  # USAC asserts when no model is found
        return 0
    return int(mask.sum()) if mask is not None else 0


rooms_by_pair = {}
for A, B, si, variant, ra, dd, rb in plan():
    e = rooms_by_pair.setdefault((A, B), {"A": set(), "B": set(), "door": dd})
    e["A"].update(ra); e["B"].update(rb)
    if len(dd) > len(e["door"]):
        e["door"] = dd

out = []
for (A, B), e in rooms_by_pair.items():
    others = [int(r) for r in SETS["rooms"] if int(r) not in (A, B)]
    ctrl_rows = {o: draw_subsets(SETS["rooms"][str(o)], N_PER_ROOM, 1, seed=19000 + o)[0] for o in others}
    rec = {"pair": [A, B], "door_rows": e["door"], "A_rows": sorted(e["A"]), "B_rows": sorted(e["B"])}
    for d in e["door"]:
        rec[f"door{d}_vs_B"] = [inliers(d, r) for r in sorted(e["B"])]
        rec[f"door{d}_vs_A"] = [inliers(d, r) for r in sorted(e["A"])]
        rec[f"door{d}_vs_control"] = {o: [inliers(d, r) for r in rows] for o, rows in ctrl_rows.items()}
    ab = {f"{a}-{b}": inliers(a, b) for a, b in product(sorted(e["A"]), sorted(e["B"]))}
    rec["A_vs_B_max"] = max(ab.values()); rec["A_vs_B_top3"] = sorted(ab.items(), key=lambda x: -x[1])[:3]
    ctrl_all = [v for d in e["door"] for vs in rec[f"door{d}_vs_control"].values() for v in vs]
    rec["control_max"] = max(ctrl_all); rec["control_p90"] = float(np.percentile(ctrl_all, 90))
    print(A, B, {d: (max(rec[f"door{d}_vs_B"]), max(rec[f"door{d}_vs_A"])) for d in e["door"]},
          "A_vs_B max", rec["A_vs_B_max"], "control max", rec["control_max"], flush=True)
    out.append(rec)
(HERE / "out/sift_links.json").write_text(json.dumps(out, indent=1))

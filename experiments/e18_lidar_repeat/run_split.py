"""LiDAR repeatability study: same room from two DISJOINT observation sets, per variant.

Split per room: visits sorted by time, A = even-indexed, B = odd-indexed; a room with one visit is
split into its first and second half in time (marked). Each half computes wall existence,
positions and topology on its own. GT-free consistency result (not the repeatability gate, which
needs two separate captures made per our protocol).
Gate reference: per wall |dA - dB| <= max(1 cm, 0.5 %).
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).parent))
import roomgeo as RG
from roomscan import geometry as G
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e18_lidar_repeat/out"; OUT.mkdir(parents=True, exist_ok=True)
NAME = sys.argv[1] if len(sys.argv) > 1 else "single_scan_with_ceiling"
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["V0", "V1", "V2", "V3"]
SPLIT = sys.argv[3] if len(sys.argv) > 3 else "chunk"   # "chunk" (method noise) or "visit"
CHUNK_S = 2.0

def tol(L): return max(0.01, 0.005 * L)

def compare(a, b):
    r = {"n_walls": [a["n_walls"], b["n_walls"]], "topology_equal": a["n_walls"] == b["n_walls"]}
    for k in ("extent_u", "extent_v"):
        d = abs(a[k] - b[k]); r[k + "_diff_mm"] = round(1000 * d, 1); r[k + "_pass"] = d <= tol(0.5 * (a[k] + b[k]))
    if a["ceiling_h"] and b["ceiling_h"]:
        r["ceiling_diff_mm"] = round(1000 * abs(a["ceiling_h"] - b["ceiling_h"]), 1)
    if r["topology_equal"]:
        la, lb = a["lengths"], b["lengths"]; n = len(la); best = None
        for direc in (1, -1):
            lbd = lb[::direc]
            for s in range(n):
                d = np.abs(la - np.roll(lbd, s)); c = d.sum()
                if best is None or c < best[0]: best = (c, d, np.roll(lbd, s))
        d, lbm = best[1], best[2]
        r["wall_diffs_mm"] = (1000 * d).round(1).tolist()
        r["wall_pass"] = [bool(x <= tol(0.5 * (p + q))) for x, p, q in zip(d, la, lbm)]
    return r

t0 = time.time()
cap = StrayCapture.load(ROOT / NAME); T = cap.T_wc(); t = cap.timestamps
pc = G.fuse(cap, T, range(0, len(cap), 5)); seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals))
lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
rof = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])
results = {}
for label in range(1, seg.labels.max() + 1):
    sp = sorted(RG.visits(rof, t, label), key=lambda ab: t[ab[0]])
    if not sp: continue
    if SPLIT == "visit" and len(sp) >= 2:
        A, B, kind = sp[0::2], sp[1::2], "visit_split"
    elif SPLIT == "visit":
        a, b = sp[0]; m = (a + b) // 2; A, B, kind = [(a, m)], [(m + 1, b)], "time_split"
    else:
        # chunk split: every visit cut into CHUNK_S chunks alternating between halves; each half keeps
        # chunks grouped by visit (alignment unit = visit)
        A, B, k = [], [], 0
        for a, b in sp:
            ga, gb = [], []
            s0 = a
            while s0 <= b:
                e0 = min(int(np.searchsorted(t, t[s0] + CHUNK_S)), b + 1) - 1
                (ga if k % 2 == 0 else gb).append((s0, max(e0, s0))); k += 1; s0 = e0 + 1
            if ga: A.append(ga)
            if gb: B.append(gb)
        kind = f"chunk_split_{CHUNK_S}s"
    dur = lambda S: round(float(sum(t[y] - t[x] for g in S for x, y in (g if isinstance(g, list) else [g]))), 1)
    results[label] = {"split": kind, "n_visits": len(sp), "dur_A_s": dur(A), "dur_B_s": dur(B)}
    for v in VARIANTS:
        ga, ia = RG.room_geometry(cap, T, seg, label, A, v); gb, ib = RG.room_geometry(cap, T, seg, label, B, v)
        if ga is None or gb is None:
            results[label][v] = {"failed": [None if ga else ia, None if gb else ib]}
        else:
            results[label][v] = {**compare(ga, gb), "align_A": ia.get("align"), "align_B": ib.get("align")}
        print(NAME, label, v, json.dumps({k: x for k, x in results[label][v].items() if not k.startswith("align")}), flush=True)
summary = {}
for v in VARIANTS:
    rows = [r[v] for r in results.values() if v in r and "failed" not in r[v]]
    fails = sum(1 for r in results.values() if v in r and "failed" in r[v])
    ext = [x for r in rows for x in (r["extent_u_diff_mm"], r["extent_v_diff_mm"])]
    extp = [x for r in rows for x in (r["extent_u_pass"], r["extent_v_pass"])]
    wd = [x for r in rows if "wall_diffs_mm" in r for x in r["wall_diffs_mm"]]
    wp = [x for r in rows if "wall_pass" in r for x in r["wall_pass"]]
    cd = [r["ceiling_diff_mm"] for r in rows if "ceiling_diff_mm" in r]
    summary[v] = {"rooms_compared": len(rows), "rooms_failed": fails,
                  "topology_equal": f"{sum(r['topology_equal'] for r in rows)}/{len(rows)}",
                  "extent_diff_mm_median": float(np.median(ext)) if ext else None, "extent_diff_mm_p90": float(np.percentile(ext, 90)) if ext else None,
                  "extent_pass_frac": float(np.mean(extp)) if extp else None,
                  "wall_diff_mm_median": float(np.median(wd)) if wd else None, "wall_diff_mm_p90": float(np.percentile(wd, 90)) if wd else None,
                  "wall_pass_frac_matched": float(np.mean(wp)) if wp else None, "n_walls_matched": len(wp),
                  "ceiling_diff_mm_median": float(np.median(cd)) if cd else None, "ceiling_diff_mm_max": float(np.max(cd)) if cd else None}
out = {"label": "GT-free within-capture split consistency (not the repeatability gate)", "capture": NAME, "sec": round(time.time() - t0, 1), "summary": summary,
       "rooms": {str(k): {kk: vv for kk, vv in v.items()} for k, v in results.items()}}
(OUT / f"split_{SPLIT}_{NAME}_{'_'.join(VARIANTS)}.json").write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
print(json.dumps(summary, indent=1))

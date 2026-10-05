"""Wall-position repeatability given topology (GT-free), per variant.

Separates the two ingredients of LiDAR repeatability:
  topology  (which walls exist)        -> from ALL data of the room, per variant
  positions (where each wall face is)  -> from each disjoint half independently
Halves = 2 s chunks alternating within every visit (both halves see every visit, share no frame).
Each half's cloud is ICP-aligned onto the full room cloud (removes rigid drift, as a per-room
alignment between two captures would), then every wall is re-measured with the variant's rule.
Gate reference: |dA - dB| <= max(1 cm, 0.5 %) per wall; ceiling spread <= 1 cm.
"""
import json, sys, time
from pathlib import Path
import numpy as np, open3d as o3d
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).parent))
import roomgeo as RG
from roomscan import geometry as G
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e18_lidar_repeat/out"
NAME = sys.argv[1] if len(sys.argv) > 1 else "single_scan_with_ceiling"
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["V0", "V1", "V2", "V3"]
CHUNK_S = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
reg = o3d.pipelines.registration
tol = lambda L: max(0.01, 0.005 * L)

def chunk_halves(sp, t):
    A, B, k = [], [], 0
    for a, b in sp:
        ga, gb, s0 = [], [], a
        while s0 <= b:
            e0 = max(min(int(np.searchsorted(t, t[s0] + CHUNK_S)), b + 1) - 1, s0)
            (ga if k % 2 == 0 else gb).append((s0, e0)); k += 1; s0 = e0 + 1
        if ga: A.append(ga)
        if gb: B.append(gb)
    return A, B

def to_full(hc, full):
    if not full.has_normals(): full.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
    r = reg.registration_icp(hc, full, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane())
    r = reg.registration_icp(hc, full, 0.02, r.transformation, reg.TransformationEstimationPointToPlane())
    h = o3d.geometry.PointCloud(hc); h.transform(r.transformation); return h, r.fitness

t0 = time.time()
cap = StrayCapture.load(ROOT / NAME); T = cap.T_wc(); t = cap.timestamps
pc = G.fuse(cap, T, range(0, len(cap), 5)); seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals))
lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
rof = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])
res = {}
for label in range(1, seg.labels.max() + 1):
    sp = sorted(RG.visits(rof, t, label), key=lambda ab: t[ab[0]])
    if not sp: continue
    A, B = chunk_halves(sp, t); res[label] = {"n_visits": len(sp)}
    for v in VARIANTS:
        # V6 = V2 topology/ceiling + wall positions from V5 (close-range) clouds, per-wall fallback to V2
        g, info = RG.room_geometry(cap, T, seg, label, sp, "V2" if v == "V6" else v)
        if g is None:
            res[label][v] = {"full_topology_failed": info}; print(label, v, "full failed", info, flush=True); continue
        full = RG.room_cloud(cap, T, seg, label, sp, "V2" if v == "V6" else v); Rm = G._rot(g["yaw"])
        row = {"n_walls": g["n_walls"], "full_lengths_m": np.round(g["lengths"], 4).tolist(), "full_ceiling_m": g["ceiling_h"]}
        halves = []
        for H_ in (A, B):
            hc, fit = to_full(RG.room_cloud(cap, T, seg, label, H_, "V2" if v == "V6" else v), full)
            P, N = np.asarray(hc.points), np.asarray(hc.normals)
            UV, NUV, Hh = P[:, [0, 2]] @ Rm.T, N[:, [0, 2]] @ Rm.T, P[:, 1] - seg.grid.floor_y
            Q, sup = RG.edge_positions(g["poly_uv"], UV, NUV, Hh, g["ceiling_h"], "V2" if v == "V6" else v)
            if v == "V6":
                cc = RG.room_cloud(cap, T, seg, label, H_, "V5")
                cc.transform(np.eye(4)); cc, _ = to_full(cc, full)
                CP, CN = np.asarray(cc.points), np.asarray(cc.normals)
                Qc, supc = RG.edge_positions(g["poly_uv"], CP[:, [0, 2]] @ Rm.T, CN[:, [0, 2]] @ Rm.T, CP[:, 1] - seg.grid.floor_y, g["ceiling_h"], "V5")
                n = len(Q); Qm = Q.copy()
                for k in range(n):                       # per-wall choice: close-range if supported
                    if supc[k] >= 60:
                        a, b = g["poly_uv"][k], g["poly_uv"][(k + 1) % n]
                        ax = 0 if abs(b[0] - a[0]) < 1e-9 else 1
                        Qm[k][ax] = Qc[k][ax]; Qm[(k + 1) % n][ax] = Qc[k][ax]
                Q, sup = Qm, [max(x, y) for x, y in zip(sup, supc)]
            fl, ce, _ = RG.floor_ceiling_robust(P, N, g["poly_uv"], Rm, seg.grid.floor_y)
            halves.append({"lengths": np.linalg.norm(np.roll(Q, -1, 0) - Q, axis=1), "support": sup, "fit": fit,
                           "ceiling": (ce - fl) if (ce is not None and fl is not None) else None})
        la, lb = halves[0]["lengths"], halves[1]["lengths"]
        d = np.abs(la - lb)
        row.update({"wall_diffs_mm": (1000 * d).round(1).tolist(), "wall_pass": [bool(x <= tol(0.5 * (p + q))) for x, p, q in zip(d, la, lb)],
                    "low_support_walls": int(sum(1 for h in halves for s in h["support"] if s < 30)), "icp_fitness": [round(h["fit"], 3) for h in halves],
                    "ceiling_diff_mm": round(1000 * abs(halves[0]["ceiling"] - halves[1]["ceiling"]), 1) if halves[0]["ceiling"] and halves[1]["ceiling"] else None})
        res[label][v] = row
        print(label, v, json.dumps({k: row[k] for k in ("n_walls", "wall_diffs_mm", "low_support_walls", "ceiling_diff_mm")}), flush=True)
summary = {}
for v in VARIANTS:
    rows = [r[v] for r in res.values() if v in r and "wall_diffs_mm" in r[v]]
    wd = [x for r in rows for x in r["wall_diffs_mm"]]; wp = [x for r in rows for x in r["wall_pass"]]
    cd = [r["ceiling_diff_mm"] for r in rows if r["ceiling_diff_mm"] is not None]
    summary[v] = {"rooms": len(rows), "rooms_full_failed": sum(1 for r in res.values() if v in r and "full_topology_failed" in r[v]),
                  "n_walls_total": len(wd), "walls_per_room": [r["n_walls"] for r in rows],
                  "wall_diff_mm_median": round(float(np.median(wd)), 1) if wd else None, "wall_diff_mm_p90": round(float(np.percentile(wd, 90)), 1) if wd else None,
                  "wall_pass_frac": round(float(np.mean(wp)), 3) if wp else None,
                  "ceiling_diff_mm_median": round(float(np.median(cd)), 1) if cd else None, "ceiling_diff_mm_max": round(float(np.max(cd)), 1) if cd else None,
                  "ceiling_pass_frac_1cm": round(float(np.mean([c <= 10 for c in cd])), 3) if cd else None}
out = {"label": "GT-free: wall-position repeatability given topology (chunk halves, per-room ICP to full)", "capture": NAME, "chunk_s": CHUNK_S,
       "sec": round(time.time() - t0, 1), "summary": summary, "rooms": {str(k): v for k, v in res.items()}}
(OUT / f"fixed_topology_{NAME}_c{CHUNK_S:g}_{'_'.join(VARIANTS)}.json").write_text(json.dumps(out, indent=1, default=float)); print(json.dumps(summary, indent=1))

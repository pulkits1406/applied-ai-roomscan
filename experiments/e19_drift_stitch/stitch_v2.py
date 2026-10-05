"""Drift ablation v2: which stitch ingredients actually reduce placement disagreement?

v1 (stitch.py) forced one common wall thickness on every shared wall and found no doorway
constraints; it moved room 4 by ~19 cm in both halves and worsened the max disagreement. Here the
ingredients are separated, all evaluated with the same GT-free metric (per-room centroid distance
between the two visit-split halves after a best rigid 2-D fit of the whole layout):
  A   raw ARKit placement
  B1  common Manhattan yaw only (rotate each room about its centroid)
  B2  B1 + doorway constraints: the jamb-to-jamb gap each room measures in its OWN wall face
      (lidar.openings.measure) must line up along the wall
  B3  B2 + shared walls with ONE common thickness (v1 assumption)
  B4  B2 + shared walls with per-wall thickness = that wall's mean raw gap over both halves'
      neighbours... not available without peeking -> per-wall thickness prior = the half's own
      raw gap, Huber-weighted (corrects only gross inconsistencies)
The shared-wall gap spread is NOT an evaluation metric (it is what B3 optimises).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import stitch as S  # noqa: E402
from roomscan import geometry as G  # noqa: E402
from roomscan.lidar import openings as OP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/e19_drift_stitch/out"


def snapped_points(P, N, raw_poly, turn):
    c = raw_poly.mean(0)
    P2, N2 = P.copy(), N.copy()
    P2[:, [0, 2]] = (P[:, [0, 2]] - c) @ S.rot2(-turn).T + c
    N2[:, [0, 2]] = N[:, [0, 2]] @ S.rot2(-turn).T
    return P2, N2


def door_constraints(snapped, clouds_snapped, doorways, yaw, floor_y):
    """For each doorway between two present rooms, each room's own jamb-gap centre along the wall."""
    Rm = G._rot(yaw); out = []
    for d in doorways:
        a, b = d.rooms
        if a not in snapped or b not in snapped:
            continue
        ms = []
        for r in (a, b):
            P, N = clouds_snapped[r]
            ms.append(OP.measure(snapped[r] @ Rm.T, Rm, d.centre_xz, P, N, floor_y))
        if any(m is None or m.centre_along is None for m in ms):
            continue
        pa = snapped[a] @ Rm.T
        ea = ms[0].edge
        ax = 0 if abs(pa[(ea + 1) % len(pa)][0] - pa[ea][0]) < 1e-9 else 1
        out.append({"a": a, "b": b, "axis": ax, "centre_a": ms[0].centre_along, "centre_b": ms[1].centre_along,
                    "width_a": ms[0].width, "width_b": ms[1].width})
    return out


def solve(rooms, walls, doors, fixed, mode):
    keys = sorted(rooms); idx = {r: i for i, r in enumerate(keys)}
    n = 2 * len(keys) + (1 if mode == "B3" else 0)
    rows, rhs, wts = [], [], []
    for d in doors:
        row = np.zeros(n); o = 1 - d["axis"]
        row[2 * idx[d["b"]] + o] += 1; row[2 * idx[d["a"]] + o] -= 1
        rows.append(row); rhs.append(d["centre_a"] - d["centre_b"]); wts.append(3.0)
    if mode in ("B3", "B4"):
        for w in walls:
            row = np.zeros(n); ax = w["axis"]
            row[2 * idx[w["b"]] + ax] += 1; row[2 * idx[w["a"]] + ax] -= 1
            if mode == "B3":
                row[-1] = w["sign_a"]; rhs.append(w["coord_a"] - w["coord_b"])
            else:  # per-wall thickness = own raw gap -> constraint only resists change of that gap
                rhs.append(w["coord_a"] - w["coord_b"] - (-w["sign_a"]) * w["gap"])
            rows.append(row); wts.append(1.0)
    for ax in (0, 1):
        row = np.zeros(n); row[2 * idx[fixed] + ax] = 1; rows.append(row); rhs.append(0.0); wts.append(10.0)
    for r in keys:
        for ax in (0, 1):
            row = np.zeros(n); row[2 * idx[r] + ax] = 1; rows.append(row); rhs.append(0.0); wts.append(0.05)
    A, b, w = np.array(rows), np.array(rhs), np.array(wts)
    x = np.zeros(n)
    for _ in range(10):                       # IRLS with Huber (k = 3 cm) on non-gauge rows
        r = A @ x - b
        hub = np.where(np.abs(r) < 0.03, 1.0, 0.03 / np.maximum(np.abs(r), 1e-9))
        W = np.sqrt(w * hub)
        x, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
    return {r: x[2 * idx[r]:2 * idx[r] + 2] for r in keys}


def corner_disagreement(PA, PB):
    """All room corners (same shapes -> same vertex order), best rigid 2-D fit of the whole layout;
    sensitive to per-room rotation, unlike the centroid metric (B1 == A under centroids)."""
    common = sorted(set(PA) & set(PB))
    X = np.concatenate([PA[r] for r in common]); Y = np.concatenate([PB[r] for r in common])
    mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((Y - my).T @ (X - mx)); D = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    d = np.linalg.norm(Y - ((X - mx) @ (U @ D @ Vt).T + my), axis=1)
    return {"median_mm": round(1000 * float(np.median(d)), 1), "p90_mm": round(1000 * float(np.percentile(d, 90)), 1), "max_mm": round(1000 * float(d.max()), 1)}


def main(name="single_scan_with_ceiling", variant="V2"):
    from scipy import ndimage as ndi
    import roomgeo as RG
    from roomscan.stray import StrayCapture
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    pc = G.fuse(cap, T, range(0, len(cap), 5)); seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals))
    doorways = OP.detect(seg)
    lab_d = ndi.grey_dilation(seg.labels, size=(7, 7))
    cij = np.clip(seg.grid.cell(T[:, [0, 2], 3]), 0, np.array(seg.grid.shape) - 1)
    rof = np.where(seg.labels[cij[:, 0], cij[:, 1]] > 0, seg.labels[cij[:, 0], cij[:, 1]], lab_d[cij[:, 0], cij[:, 1]])
    halves, spans_all = {"A": {}, "B": {}}, {}
    for label in range(1, seg.labels.max() + 1):
        sp = sorted(RG.visits(rof, t, label), key=lambda ab: t[ab[0]])
        if sp:
            spans_all[label] = sp
        if len(sp) >= 2:
            halves["A"][label], halves["B"][label] = sp[0::2], sp[1::2]
    geo_full, clouds_full = S.full_rooms(cap, T, seg, spans_all, variant)
    layouts = {m: {} for m in ("A", "B1", "B2", "B3", "B4")}; diag = {}
    for h, spans in halves.items():
        geo, clouds, _ = S.place_half(cap, T, seg, geo_full, clouds_full, spans, variant)
        raw = {r: S.world_poly(g) for r, g in geo.items()}
        yaw = S.common_yaw([g["yaw"] for g in geo.values()])
        snapped, csn = {}, {}
        for r, g in geo.items():
            snapped[r], turn = S.snap_yaw(raw[r], g["yaw"], yaw)
            csn[r] = snapped_points(*clouds[r], raw[r], turn)
        walls = S.shared_walls(snapped, yaw)
        doors = door_constraints(snapped, csn, doorways, yaw, seg.grid.floor_y)
        fixed = max(geo, key=lambda r: geo[r]["area"]); Rm = G._rot(yaw)
        layouts["A"][h] = raw; layouts["B1"][h] = snapped
        for m in ("B2", "B3", "B4"):
            tr = solve(snapped, walls, doors, fixed, m)
            layouts[m][h] = {r: snapped[r] + tr[r] @ Rm for r in snapped}
        diag[h] = {"doorways_detected": len(doorways), "door_constraints": doors, "n_shared_walls": len(walls)}
    res = {"label": "GT-free drift ablation v2 (visit-split halves, shared room shapes)", "capture": name, "variant": variant, "diagnostics": diag}
    for m, L in layouts.items():
        per, med, mx = S.rigid_disagreement(L["A"], L["B"])
        res[m] = {"per_room_mm": per, "median_mm": med, "max_mm": mx, "corners": corner_disagreement(L["A"], L["B"])}
    (OUT / f"ablation_v2_{name}_{variant}.json").write_text(json.dumps(res, indent=1, default=float))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))
    for ax, m in zip(axs, ("A", "B2", "B4")):
        for h, col in (("A", "tab:blue"), ("B", "tab:red")):
            for r, P in layouts[m][h].items():
                Q = np.vstack([P, P[:1]]); ax.plot(Q[:, 0], Q[:, 1], color=col, lw=1.2)
                if h == "A":
                    ax.text(*P.mean(0), str(r))
        ax.set_aspect("equal"); ax.set_title(f"{m}: halves A (blue) / B (red); median {res[m]['median_mm']} mm, max {res[m]['max_mm']} mm")
    fig.tight_layout(); fig.savefig(OUT / f"footprint_v2_{name}_{variant}.png", dpi=75)
    print(json.dumps({m: {k: res[m][k] for k in ("median_mm", "max_mm", "corners")} for m in layouts}, indent=1))
    print(json.dumps({h: {"doors": len(diag[h]["door_constraints"]), "walls": diag[h]["n_shared_walls"], "detected": diag[h]["doorways_detected"]} for h in diag}))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e18_lidar_repeat"))
    main(*(sys.argv[1:] or []))

"""Within-capture drift: depth reprojection residual for revisit pairs vs time gap.

If VIO poses drift, two frames that view the same surface far apart in time disagree by more
than the short-baseline noise floor (~6-8 mm, e01). ARKit relocalisation can also *snap*
poses, which shows up as a step rather than a ramp.
"""
import json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e03_drift/out"; OUT.mkdir(parents=True, exist_ok=True)

def residual(cap, i, j, Ts):
    d = cap.depth_m(i); c = cap.confidence(i); K = cap.K_depth(i)
    v, u = np.nonzero((c == 2) & (d < 4.0)); z = d[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    pw = pc @ Ts[i, :3, :3].T + Ts[i, :3, 3]
    pj = (pw - Ts[j, :3, 3]) @ Ts[j, :3, :3]
    Kj = cap.K_depth(j); m = pj[:, 2] > 0.1; pj = pj[m]
    uj = Kj[0, 0] * pj[:, 0] / pj[:, 2] + Kj[0, 2]; vj = Kj[1, 1] * pj[:, 1] / pj[:, 2] + Kj[1, 2]
    inb = (uj >= 0) & (uj < 255.5) & (vj >= 0) & (vj < 191.5)
    if inb.sum() < 2000: return None
    ui, vi = np.round(uj[inb]).astype(int), np.round(vj[inb]).astype(int)
    dj, cj = cap.depth_m(j), cap.confidence(j)
    ok = cj[vi, ui] == 2
    r = dj[vi, ui][ok] - pj[inb][ok, 2]
    r = r[np.abs(r) < 0.3]  # exclude occlusion/disocclusion
    return float(np.median(np.abs(r))) if len(r) > 1000 else None

res = {}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, name in zip(axs, ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]):
    cap = StrayCapture.load(ROOT / name); Ts = cap.T_wc(); t = cap.timestamps
    pos = Ts[:, :3, 3]; fwd = Ts[:, :3, 2]
    idx = np.arange(0, len(cap), 6)
    pairs = []
    for a in idx:
        b = idx[idx > a + 60]
        ok = (np.linalg.norm(pos[b] - pos[a], axis=1) < 0.4) & (fwd[b] @ fwd[a] > np.cos(np.radians(20)))
        pairs += [(a, j) for j in b[ok]]
    rng = np.random.default_rng(0)
    if len(pairs) > 400: pairs = [pairs[k] for k in rng.choice(len(pairs), 400, replace=False)]
    rows = [(t[j] - t[i], residual(cap, i, j, Ts)) for i, j in pairs]
    rows = np.array([r for r in rows if r[1] is not None])
    bins = [0, 2, 10, 30, 60, 120, 1e9]
    stats = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = rows[(rows[:, 0] >= lo) & (rows[:, 0] < hi), 1]
        if len(s): stats[f"{lo}-{hi if hi < 1e9 else 'inf'}s"] = {"n": int(len(s)), "median_mm": round(1000 * float(np.median(s)), 1), "p90_mm": round(1000 * float(np.percentile(s, 90)), 1)}
    res[name] = stats; print(name, json.dumps(stats))
    ax.scatter(rows[:, 0], 1000 * rows[:, 1], s=4); ax.set_xlabel("time gap (s)"); ax.set_ylabel("median |depth resid| (mm)")
    ax.set_title(name); ax.set_ylim(0, 80)
fig.tight_layout(); fig.savefig(OUT / "revisit_residual.png", dpi=90)
(OUT / "revisit_residual.json").write_text(json.dumps(res, indent=1))

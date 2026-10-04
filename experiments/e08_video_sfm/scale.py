"""Metric scale for an RGB-only SfM model from per-frame metric depth (MoGe-2, or LiDAR as an oracle).

For each registered keyframe, every observed sparse 3D point gives a ratio
    r = metric_depth(u, v) / z_sfm,   z_sfm = depth of the point in that camera (SfM units),
an estimate of metres per SfM unit. The reference scale is the Sim(3) (Umeyama) scale aligning SfM
camera centres to ARKit centres (pseudo-GT). Estimators:
    pooled    = median of all ratios of the selected frames
    per_frame = median over frames of the per-frame median ratio

Usage: uv run python experiments/e08_video_sfm/scale.py <capture> <cfg_dir_name> <variant> [--max-frames 250]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from common import CACHE, FRAMES, OUT, ROOT, load_frames, umeyama, traj_metrics
from sfm import cam_from_world

N_SUBSETS = [5, 10, 25, 50, 100, 200]
REPEATS = 20


def load_models(model_root):
    recs = []
    for d in sorted(p for p in model_root.iterdir() if p.is_dir()):
        if (d / "images.bin").exists() or (d / "images.txt").exists():
            recs.append(pycolmap.Reconstruction(d))
    return sorted(recs, key=lambda r: -r.num_reg_images())


def gravity_rotation(T_wc):
    """Which rotation makes the image upright according to the ARKit gravity direction (diagnostic only)."""
    up = T_wc[:3, :3].T @ np.array([0, 1.0, 0])
    ux, uy = up[0], up[1]
    if -uy >= abs(ux):
        return "none"
    if uy >= abs(ux):
        return "180"
    return "ccw" if ux > 0 else "cw"


def moge_depth(capture, rows):
    """Load cached MoGe depth, computing missing rows in a subprocess (see moge_depth.py)."""
    d = CACHE / "moge" / capture
    d.mkdir(parents=True, exist_ok=True)
    rows_p = d / "_request.json"
    rows_p.write_text(json.dumps([int(r) for r in rows]))
    here = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(here / "moge_depth.py"), capture, str(rows_p)], check=True, cwd=here)
    return {r: np.load(d / f"{r:06d}.npy").astype(np.float32) for r in rows}


def lidar_depth(capture, row):
    d = cv2.imread(str(ROOT / capture / "depth" / f"{row:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    c = cv2.imread(str(ROOT / capture / "confidence" / f"{row:06d}.png"), cv2.IMREAD_UNCHANGED)
    d[(c != 2) | (d <= 0)] = np.nan
    # Keypoints cluster on depth edges where the coarse LiDAR map mixes foreground and background;
    # keep only pixels whose 5x5 neighbourhood is depth-smooth (max/min within 3%).
    filled = np.where(np.isfinite(d), d, 0).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    hi = cv2.dilate(filled, k)
    lo = cv2.erode(np.where(np.isfinite(d), d, 1e3).astype(np.float32), k)
    d[hi > 1.03 * lo] = np.nan
    return d


def sample(depth, uv, wh=(960, 720)):
    h, w = depth.shape
    x = np.clip(np.round(uv[:, 0] * w / wh[0] - 0.5).astype(int), 0, w - 1)
    y = np.clip(np.round(uv[:, 1] * h / wh[1] - 0.5).astype(int), 0, h - 1)
    return depth[y, x]


def frame_observations(rec, img):
    """(uv, z_sfm, point3D_ids) of well-constrained 3D points observed in this image."""
    T = cam_from_world(img)
    uv, X, ids = [], [], []
    for p2 in img.points2D:
        if not p2.has_point3D():
            continue
        p3 = rec.points3D[p2.point3D_id]
        if p3.track.length() < 3 or p3.error > 2.0:
            continue
        uv.append(p2.xy)
        X.append(p3.xyz)
        ids.append(p2.point3D_id)
    if not uv:
        return np.zeros((0, 2)), np.zeros(0), []
    X = np.array(X)
    z = (T[:3, :3] @ X.T + T[:3, 3:4])[2]
    return np.array(uv), z, ids


def arkit_triangulate(rec, frames, pids):
    """Linear (DLT) triangulation of SfM tracks using ARKit poses and true K: metric points that do not
    depend on the SfM poses. Returns {point3D_id: xyz in ARKit world}."""
    P = {}
    for iid, im in rec.images.items():
        if im.has_pose:
            f = frames[int(im.name[:6])]
            P[iid] = np.array(f["K"]) @ np.linalg.inv(np.array(f["T_wc"]))[:3]
    out = {}
    for pid in pids:
        A = []
        for el in rec.points3D[pid].track.elements:
            if el.image_id in P:
                x, y = rec.images[el.image_id].points2D[el.point2D_idx].xy
                Pi = P[el.image_id]
                A += [x * Pi[2] - Pi[0], y * Pi[2] - Pi[1]]
        if len(A) >= 6:
            _, _, Vt = np.linalg.svd(np.array(A))
            X = Vt[-1]
            if abs(X[3]) > 1e-12:
                out[pid] = X[:3] / X[3]
    return out


def estimators(ratios_per_frame, idx):
    sel = [ratios_per_frame[i] for i in idx]
    pooled = float(np.median(np.concatenate(sel)))
    per_frame = float(np.median([np.median(r) for r in sel]))
    return pooled, per_frame


def convergence(ratios_per_frame, s_true, rng):
    n = len(ratios_per_frame)
    res = {}
    for N in N_SUBSETS:
        if N > n:
            continue
        errs = {"pooled": [], "per_frame": []}
        for _ in range(REPEATS if N < n else 1):
            idx = rng.choice(n, N, replace=False)
            a, b = estimators(ratios_per_frame, idx)
            errs["pooled"].append(a / s_true - 1)
            errs["per_frame"].append(b / s_true - 1)
        res[N] = {k: {"median_abs_pct": float(100 * np.median(np.abs(v))), "p90_abs_pct": float(100 * np.percentile(np.abs(v), 90)),
                      "mean_signed_pct": float(100 * np.mean(v))} for k, v in errs.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("cfg")
    ap.add_argument("variant")
    ap.add_argument("--max-frames", type=int, default=250)
    ap.add_argument("--min-model", type=int, default=30)
    ap.add_argument("--per-model", type=int, default=10**6, help="max frames sampled per model")
    a = ap.parse_args()
    frames = {f["row"]: f for f in load_frames(a.capture)}
    models = load_models(CACHE / a.cfg / a.variant)
    rng = np.random.default_rng(0)
    out = {"capture": a.capture, "cfg": a.cfg, "variant": a.variant, "models": [],
           "note": "reference scale = Sim3(SfM->ARKit) pseudo-GT; LiDAR = ARKit LiDAR pseudo-GT depth"}
    budget = a.max_frames
    plan = []
    for mi, rec in enumerate(models):
        imgs = sorted([im for im in rec.images.values() if im.has_pose], key=lambda im: im.name)
        if len(imgs) < a.min_model or budget <= 0:
            continue
        use = np.linspace(0, len(imgs) - 1, min(len(imgs), budget, a.per_model)).round().astype(int)
        budget -= len(use)
        plan.append((mi, rec, imgs, use))
    t0 = time.time()
    moge = moge_depth(a.capture, sorted({int(imgs[k].name[:6]) for _, _, imgs, use in plan for k in use}))
    out["sec_moge_total"] = time.time() - t0
    rot_stats = {}
    norm_moge, norm_lidar = [], []
    for mi, rec, imgs, use in plan:
        T_sfm = np.stack([np.linalg.inv(cam_from_world(im)) for im in imgs])
        rows = [int(im.name[:6]) for im in imgs]
        T_gt = np.stack([np.array(frames[r]["T_wc"]) for r in rows])
        s_true, _, _ = umeyama(T_sfm[:, :3, 3], T_gt[:, :3, 3])
        r_moge, r_lidar, npts, rows_ok = [], [], [], []
        tri_lidar, tri_sfm = [], []
        obs = {k: frame_observations(rec, imgs[k]) for k in use}
        Xa = arkit_triangulate(rec, frames, {pid for k in use for pid in obs[k][2]})
        for k in use:
            im, row = imgs[k], rows[k]
            g = gravity_rotation(T_gt[k])
            rot_stats[g] = rot_stats.get(g, 0) + 1
            uv, z, ids = obs[k]
            if len(z) < 10:
                continue
            dm = sample(moge[row], uv)
            dl = sample(lidar_depth(a.capture, row), uv)
            vm = np.isfinite(dm) & (dm > 0) & (z > 0)
            vl = np.isfinite(dl) & (dl > 0) & (z > 0) & vm
            if vm.sum() >= 10 and vl.sum() >= 10:
                r_moge.append(dm[vm] / z[vm])
                r_lidar.append(dl[vl] / z[vl])
                npts.append(int(vm.sum()))
                rows_ok.append(row)
                # ARKit-triangulated depth of the same points: LiDAR consistency and SfM structure scale.
                Tcw = np.linalg.inv(T_gt[k])
                za = np.array([(Tcw[:3, :3] @ Xa[i] + Tcw[:3, 3])[2] if i in Xa else np.nan for i in ids])
                va = vl & np.isfinite(za) & (za > 0.1) & (za < 10)
                if va.sum() >= 10:
                    tri_lidar.append(float(np.median(dl[va] / za[va])))
                    tri_sfm.append(float(np.median(za[va] / z[va]) / s_true))
            print(a.capture, mi, row, len(z), round(float(np.median(dm[vm] / z[vm]) / s_true), 3) if vm.sum() else None, flush=True)
        if len(r_moge) < 5:
            continue
        all_idx = np.arange(len(r_moge))
        m_pool, m_pf = estimators(r_moge, all_idx)
        l_pool, l_pf = estimators(r_lidar, all_idx)
        per_frame_m = np.array([np.median(r) for r in r_moge]) / s_true
        per_frame_l = np.array([np.median(r) for r in r_lidar]) / s_true
        ent = {"model": mi, "n_registered": len(imgs), "n_frames_used": len(r_moge), "median_points_per_frame": float(np.median(npts)),
               "s_true_m_per_unit": s_true,
               "moge": {"pooled_err_pct": 100 * (m_pool / s_true - 1), "per_frame_err_pct": 100 * (m_pf / s_true - 1),
                        "single_frame_abs_err_median_pct": float(100 * np.median(np.abs(per_frame_m - 1))),
                        "single_frame_abs_err_p90_pct": float(100 * np.percentile(np.abs(per_frame_m - 1), 90)),
                        "convergence": convergence(r_moge, s_true, rng)},
               "lidar": {"pooled_err_pct": 100 * (l_pool / s_true - 1), "per_frame_err_pct": 100 * (l_pf / s_true - 1),
                         "single_frame_abs_err_median_pct": float(100 * np.median(np.abs(per_frame_l - 1))),
                         "single_frame_abs_err_p90_pct": float(100 * np.percentile(np.abs(per_frame_l - 1), 90)),
                         "convergence": convergence(r_lidar, s_true, rng)},
               "moge_over_lidar_per_frame_median": float(np.median(per_frame_m / per_frame_l)),
               "arkit_tri": {"n_frames": len(tri_lidar),
                             "lidar_over_arkit_tri_median": float(np.median(tri_lidar)) if tri_lidar else None,
                             "arkit_tri_structure_scale_over_s_true_median": float(np.median(tri_sfm)) if tri_sfm else None},
               "per_frame_ratio_moge": per_frame_m.round(4).tolist(), "per_frame_ratio_lidar": per_frame_l.round(4).tolist(),
               "rows_used": rows_ok}
        out["models"].append(ent)
        norm_moge += [r / s_true for r in r_moge]
        norm_lidar += [r / s_true for r in r_lidar]
    # Ratios divided by their own model's reference scale, pooled over all models: how the estimator
    # converges with frame count if all frames belonged to one consistently scaled model.
    if norm_moge:
        out["all_models_normalized"] = {
            "n_frames": len(norm_moge),
            "moge": {"pooled_err_pct": 100 * (estimators(norm_moge, np.arange(len(norm_moge)))[0] - 1),
                     "per_frame_err_pct": 100 * (estimators(norm_moge, np.arange(len(norm_moge)))[1] - 1),
                     "convergence": convergence(norm_moge, 1.0, rng)},
            "lidar": {"pooled_err_pct": 100 * (estimators(norm_lidar, np.arange(len(norm_lidar)))[0] - 1),
                      "per_frame_err_pct": 100 * (estimators(norm_lidar, np.arange(len(norm_lidar)))[1] - 1),
                      "convergence": convergence(norm_lidar, 1.0, rng)}}
        print(json.dumps({k: v for k, v in ent.items() if not k.startswith("per_frame") and k != "rows_used"}, default=float)[:600], flush=True)
    out["gravity_upright_rotation_counts"] = rot_stats
    tp = CACHE / "moge" / a.capture / "timing.json"
    out["moge_timing"] = json.loads(tp.read_text()) if tp.exists() else None
    (OUT / f"{a.capture}_scale.json").write_text(json.dumps(out, indent=1))
    plot(out)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = out["models"][0]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for src, c in [("moge", "C0"), ("lidar", "C1")]:
        cv = m[src]["convergence"]
        Ns = sorted(int(k) for k in cv)
        for est, ls in [("pooled", "-"), ("per_frame", "--")]:
            ax[0].plot(Ns, [cv[n][est]["median_abs_pct"] for n in Ns], ls, color=c, marker="o", label=f"{src} {est} median")
            ax[0].plot(Ns, [cv[n][est]["p90_abs_pct"] for n in Ns], ls, color=c, alpha=0.4, label=f"{src} {est} p90")
    ax[0].axhline(3, color="k", lw=0.8, ls=":")
    ax[0].set_xscale("log"); ax[0].set_xlabel("frames used"); ax[0].set_ylabel("|scale error| % vs Sim3 to ARKit")
    ax[0].legend(fontsize=7); ax[0].set_title(f"{out['capture']} largest model ({m['n_registered']} reg.)")
    ax[1].plot(m["per_frame_ratio_moge"], ".", label="MoGe per-frame / s_true")
    ax[1].plot(m["per_frame_ratio_lidar"], ".", label="LiDAR per-frame / s_true")
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_xlabel("frame index (time order)"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{out['capture']}_scale_convergence.png", dpi=90)


if __name__ == "__main__":
    main()

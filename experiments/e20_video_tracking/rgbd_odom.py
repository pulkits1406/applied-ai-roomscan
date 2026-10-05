"""Pseudo-RGB-D odometry: MoGe-2 metric depth per keyframe + frame-to-frame registration (project venv, CPU).

Usage: uv run python experiments/e20_video_tracking/rgbd_odom.py <capture> --stride 2
Variants (all produce ONE trajectory by construction; failed links fall back to constant velocity):
  o3d      Open3D hybrid (photometric + geometric) RGB-D odometry, init = identity (then previous
           relative motion); results with > 45 deg or > 0.5 m per link are rejected.
  pnp      ALIKED+LightGlue matches (cache from features.py) lifted to 3D with frame i's MoGe depth,
           PnP-RANSAC into frame j. Fallback ladder per link: PnP -> PnP split through the skipped
           keyframe -> rotation-only bearing fit (+ previous translation) -> constant velocity.
  pnp_pgo  pnp edges for neighbours 1 and 2 (in stride units) + retrieved loop edges, then an Open3D
           pose-graph optimisation (loop edges marked uncertain so wrong loops get pruned).
Writes cache/e20/poses/<capture>_<variant>_s<stride>.json and out/<capture>_<variant>_s<stride>_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, MOGE_WH, OUT, evaluate_method, img_path, load_frames, moge_path, rot_deg, save_poses  # noqa: E402

MIN_INL = 20
PNP_PX = 5.0  # MoGe depth errors of a few % move reprojections by several px at 0.3-1 m
DS = 0.5  # MoGe cache resolution / full resolution
reg = o3d.pipelines.registration
odo = o3d.pipelines.odometry


class Frames:
    def __init__(self, capture: str, rows: list[int], K_full: np.ndarray):
        self.capture, self.rows, self.K = capture, rows, K_full
        Kd = K_full.copy()
        Kd[:2] *= DS
        self.Kd = Kd
        self.intr = o3d.camera.PinholeCameraIntrinsic(MOGE_WH[0], MOGE_WH[1], Kd[0, 0], Kd[1, 1], Kd[0, 2], Kd[1, 2])
        self._d, self._rgbd, self._f = {}, {}, {}

    def depth(self, r):
        if r not in self._d:
            d = np.load(moge_path(self.capture, r)).astype(np.float32)
            self._d[r] = d
            if len(self._d) > 64:
                self._d.pop(next(iter(self._d)))
        return self._d[r]

    def rgbd(self, r):
        if r not in self._rgbd:
            c = cv2.resize(cv2.cvtColor(cv2.imread(str(img_path(self.capture, r))), cv2.COLOR_BGR2RGB), MOGE_WH,
                           interpolation=cv2.INTER_AREA)
            d = np.nan_to_num(self.depth(r), nan=0.0)
            self._rgbd[r] = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(c), o3d.geometry.Image(d), depth_scale=1.0, depth_trunc=6.0,
                convert_rgb_to_intensity=True)
            if len(self._rgbd) > 16:
                self._rgbd.pop(next(iter(self._rgbd)))
        return self._rgbd[r]

    def kpts(self, r):
        if r not in self._f:
            self._f[r] = np.load(CACHE / "feat" / self.capture / f"{r:06d}.npz")["kpts"]
            if len(self._f) > 64:
                self._f.pop(next(iter(self._f)))
        return self._f[r]

    def lift(self, r, uv: np.ndarray):
        """Full-res pixels -> 3D points in camera r (MoGe depth, nearest sample); NaN where no depth."""
        d = self.depth(r)
        x = np.clip(np.round((uv[:, 0] + 0.5) * DS - 0.5).astype(int), 0, d.shape[1] - 1)
        y = np.clip(np.round((uv[:, 1] + 0.5) * DS - 0.5).astype(int), 0, d.shape[0] - 1)
        z = d[y, x]
        X = (uv[:, 0] - self.K[0, 2]) / self.K[0, 0] * z
        Y = (uv[:, 1] - self.K[1, 2]) / self.K[1, 1] * z
        return np.stack([X, Y, z], 1)


def o3d_odom(F: Frames, a: int, b: int, init: np.ndarray):
    """Returns T_b<-a (maps camera-a points into camera b) or None."""
    opt = odo.OdometryOption(iteration_number_per_pyramid_level=o3d.utility.IntVector([20, 10, 5]),
                             depth_diff_max=0.07, depth_min=0.1, depth_max=6.0)
    ok, T, info = odo.compute_rgbd_odometry(F.rgbd(a), F.rgbd(b), F.intr, init, odo.RGBDOdometryJacobianFromHybridTerm(), opt)
    return (T, info) if ok else None


def pnp(F: Frames, a: int, b: int, M: np.ndarray | None, lift: str = "a"):
    """PnP between keyframes a and b using MoGe depth of one side. Returns (T_b<-a, n_inliers, info) or None.

    lift="a": a's keypoints lifted to 3D and projected into b. lift="b": the reverse, then inverted, so a
    frame without cached depth can still be linked as long as its partner has depth.
    """
    if M is None or len(M) < MIN_INL:
        return None
    ua, ub = F.kpts(a)[M[:, 0]], F.kpts(b)[M[:, 1]]
    src_r, uv_src, uv_dst = (a, ua, ub) if lift == "a" else (b, ub, ua)
    if moge_path(F.capture, src_r) is None:
        return None
    P = F.lift(src_r, uv_src)
    v = np.isfinite(P).all(1) & (P[:, 2] > 0.05)
    if v.sum() < MIN_INL:
        return None
    ok, rv, tv, inl = cv2.solvePnPRansac(P[v], uv_dst[v].astype(np.float64), F.K, None, iterationsCount=500,
                                         reprojectionError=PNP_PX, confidence=0.999, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inl is None or len(inl) < MIN_INL:
        return None
    inl = inl[:, 0]
    rv, tv = cv2.solvePnPRefineLM(P[v][inl], uv_dst[v][inl].astype(np.float64), F.K, None, rv, tv)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rv)[0]
    T[:3, 3] = tv[:, 0]
    info = np.eye(6) * len(inl)
    dst_r = b if lift == "a" else a
    if moge_path(F.capture, dst_r) is not None:
        # Information from the 3D-3D inlier correspondences (the other frame's MoGe depth) at the PnP solution.
        Pd = F.lift(dst_r, uv_dst[v][inl])
        good = np.isfinite(Pd).all(1)
        if good.sum() >= 10:
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[v][inl][good]))
            tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pd[good]))
            info = reg.get_information_matrix_from_point_clouds(src, tgt, 0.1, T)
    if lift == "b":
        T = np.linalg.inv(T)
    return T, int(len(inl)), info


def rotation_only(F: Frames, a: int, b: int, M: np.ndarray | None, t_guess: np.ndarray):
    """Pure-rotation fit of matched bearings (2-point RANSAC + Kabsch); translation = t_guess.

    Used where PnP fails: fast turns close to a wall, where rotation dominates the image motion.
    """
    if M is None or len(M) < 10:
        return None
    Ki = np.linalg.inv(F.K)
    fa = np.c_[F.kpts(a)[M[:, 0]], np.ones(len(M))] @ Ki.T
    fb = np.c_[F.kpts(b)[M[:, 1]], np.ones(len(M))] @ Ki.T
    fa /= np.linalg.norm(fa, axis=1, keepdims=True)
    fb /= np.linalg.norm(fb, axis=1, keepdims=True)

    def kabsch(x, y):  # R with y ~= R x
        U, _, Vt = np.linalg.svd(y.T @ x)
        return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt

    rng = np.random.default_rng(0)
    best, thr = None, np.cos(np.radians(1.0))
    for _ in range(200):
        i = rng.choice(len(M), 2, replace=False)
        R = kabsch(fa[i], fb[i])
        inl = (fa @ R.T * fb).sum(1) > thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best.sum() < 10:
        return None
    T = np.eye(4)
    T[:3, :3] = kabsch(fa[best], fb[best])
    T[:3, 3] = t_guess
    return T, int(best.sum())


def link(F: Frames, a: int, b: int, mids: list[int], M: dict, rel_prev: np.ndarray):
    """T_b<-a with a fallback ladder. Returns (T, info, kind)."""
    r = pnp(F, a, b, M.get(f"{a}_{b}"))
    if r is not None:
        return r[0], r[2], "pnp"
    if mids:  # split through the intermediate keyframe(s); each half lifts from the side that has depth
        m = mids[len(mids) // 2]
        r1 = pnp(F, a, m, M.get(f"{a}_{m}"), "a")
        r2 = pnp(F, m, b, M.get(f"{m}_{b}"), "b")
        if r1 is not None and r2 is not None:
            return r2[0] @ r1[0], np.linalg.pinv(np.linalg.pinv(r1[2]) + np.linalg.pinv(r2[2])), "pnp_split"
    r = rotation_only(F, a, b, M.get(f"{a}_{b}"), rel_prev[:3, 3])
    if r is not None:
        return r[0], np.eye(6) * 1e-2, "rot_only"
    if mids:
        m = mids[len(mids) // 2]
        r1 = rotation_only(F, a, m, M.get(f"{a}_{m}"), 0.5 * rel_prev[:3, 3])
        r2 = rotation_only(F, m, b, M.get(f"{m}_{b}"), 0.5 * rel_prev[:3, 3])
        if r1 is not None and r2 is not None:
            return r2[0] @ r1[0], np.eye(6) * 1e-2, "rot_only_split"
    return rel_prev, np.eye(6) * 1e-3, "const_vel"


def load_matches(capture: str) -> dict[str, np.ndarray]:
    out = {}
    for p in sorted((CACHE / "match" / capture).glob("chunk_*.npz")):
        z = np.load(p)
        out.update({k: z[k] for k in z.files})
    return out


def chain(F: Frames, rows, M, variant: str, all_rows: list[int]):
    """Sequential odometry. Returns world poses (T_wc, first frame = identity), edge list, link stats."""
    pos = {r: i for i, r in enumerate(all_rows)}
    T_wc = [np.eye(4)]
    rel_prev = np.eye(4)
    stats: dict[str, int] = {}
    edges = []
    for k in range(1, len(rows)):
        a, b = rows[k - 1], rows[k]
        if variant == "o3d":
            # Identity init first: a wrong constant-velocity init converges to (and then propagates) a wrong
            # minimum. Results with implausible inter-frame motion are rejected.
            r = None
            for init in (np.eye(4), rel_prev):
                r = o3d_odom(F, a, b, init)
                if r is not None and rot_deg(np.eye(3), r[0][:3, :3]) < 45 and np.linalg.norm(r[0][:3, 3]) < 0.5:
                    break
                r = None
            T_ba, info, kind = (r[0], r[1], "o3d") if r is not None else (rel_prev, np.eye(6) * 1e-3, "const_vel")
        else:
            T_ba, info, kind = link(F, a, b, all_rows[pos[a] + 1:pos[b]], M, rel_prev)
        stats[kind] = stats.get(kind, 0) + 1
        edges.append((k - 1, k, T_ba, info, False, kind))
        T_wc.append(T_wc[-1] @ np.linalg.inv(T_ba))
        rel_prev = T_ba
    return T_wc, edges, stats


def pose_graph(F: Frames, rows, M, T_wc, edges, loop_pairs):
    idx = {r: i for i, r in enumerate(rows)}
    extra, n_loop_try, n_loop = [], 0, 0
    for k in range(2, len(rows)):  # skip-one neighbours
        a, b = rows[k - 2], rows[k]
        r = pnp(F, a, b, M.get(f"{a}_{b}"))
        if r is not None:
            extra.append((k - 2, k, r[0], r[2], False, "pnp2"))
    for a, b in loop_pairs:
        if a not in idx or b not in idx:
            continue
        n_loop_try += 1
        r = pnp(F, a, b, M.get(f"{a}_{b}"))
        if r is not None and r[1] >= 60:
            extra.append((idx[a], idx[b], r[0], r[2], True, "loop"))
            n_loop += 1
    pg = reg.PoseGraph()
    for T in T_wc:
        pg.nodes.append(reg.PoseGraphNode(T))
    for i, j, T, info, unc, _ in edges + extra:
        pg.edges.append(reg.PoseGraphEdge(i, j, T, info, uncertain=unc))
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.05, edge_prune_threshold=0.25,
                                       preference_loop_closure=1.0, reference_node=0)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), opt)
    kept_loops = sum(1 for e in pg.edges if e.uncertain)
    return [np.array(n.pose) for n in pg.nodes], {"loop_candidates": n_loop_try, "loop_edges": n_loop,
                                                  "loop_edges_kept": kept_loops,
                                                  "skip_edges": sum(1 for e in extra if e[5] == "pnp2")}


def link_errors(frames, rows, edges) -> dict:
    """Per-link error vs ARKit, split by the link type and by ARKit angular speed (diagnostic only)."""
    by_row = {f["row"]: f for f in frames}
    out = {}
    for i, j, T, _, _, kind in edges:
        a, b = rows[i], rows[j]
        Tg = np.linalg.inv(np.array(by_row[b]["T_wc"])) @ np.array(by_row[a]["T_wc"])
        fast = by_row[a]["ang_speed_dps"] > 80
        key = f"{kind}_{'fast' if fast else 'slow'}"
        o = out.setdefault(key, {"n": 0, "rot": [], "tlen": []})
        o["n"] += 1
        o["rot"].append(rot_deg(T[:3, :3], Tg[:3, :3]))
        if np.linalg.norm(Tg[:3, 3]) > 0.05:
            o["tlen"].append(100 * (np.linalg.norm(T[:3, 3]) / np.linalg.norm(Tg[:3, 3]) - 1))
    res = {}
    for k, o in sorted(out.items()):
        r = np.array(o["rot"])
        tl = np.array(o["tlen"])
        res[k] = {"n": o["n"], "rot_err_deg_median": float(np.median(r)), "rot_err_deg_p90": float(np.percentile(r, 90)),
                  "rot_err_deg_sum": float(r.sum()),
                  "tlen_err_pct_median": float(np.median(tl)) if len(tl) else None,
                  "tlen_abs_err_pct_median": float(np.median(np.abs(tl))) if len(tl) else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=["o3d", "pnp", "pnp_pgo"])
    a = ap.parse_args()
    frames = load_frames(a.capture)
    rows = [f["row"] for f in frames[::a.stride] if moge_path(a.capture, f["row"]) is not None]
    missing = len(frames[::a.stride]) - len(rows)
    if missing:
        print(f"WARNING {missing} stride rows lack MoGe depth; they are skipped")
    K = np.median(np.array([f["K"] for f in frames]), axis=0)
    M = load_matches(a.capture) if any(v != "o3d" for v in a.variants) else {}
    pj = json.loads((CACHE / "match" / a.capture / "pairs.json").read_text()) if M else {"loop": []}
    tim_p = CACHE / "moge" / a.capture / "timing.json"
    moge_spf = 1.75  # measured MoGe-2 vitl, default resolution, M4 MPS (e08: 1.76-1.81 s, here 1.52-1.6 s)
    for variant in a.variants:
        F = Frames(a.capture, rows, K)
        t0 = time.time()
        T_wc, edges, stats = chain(F, rows, M, variant, [f["row"] for f in frames])
        extra = {}
        if variant == "pnp_pgo":
            T_wc, extra = pose_graph(F, rows, M, T_wc, edges, [tuple(p) for p in pj["loop"]])
        sec = time.time() - t0
        extra["link_errors"] = link_errors(frames, rows, edges)
        poses = dict(zip(rows, T_wc))
        name = f"{variant}_s{a.stride}"
        save_poses(CACHE / "poses" / f"{a.capture}_{name}.json", poses, None, method=name)
        ev = evaluate_method(a.capture, rows, poses, None, metric=True,
                             extra={"method": name, "links": stats, "sec_odometry": sec, "moge_sec_per_frame": moge_spf,
                                    "missing_depth_rows": missing, **extra})
        (OUT / f"{a.capture}_{name}_eval.json").write_text(json.dumps(ev, indent=1))
        lg = ev["largest"]
        print(name, stats, extra, f"sec {sec:.1f}", "ATE", round(lg["ate_rmse_m"], 3),
              "scale_err%", round(lg["metric_scale_err_pct"], 2), "win", lg.get("win_scale_minmax"),
              "winATE", lg.get("win_ate_median_m"), "drift", lg.get("drift_fit30", {}).get("err_end_m"),
              "LR", lg.get("longrange_dist_err_pct"), flush=True)
        if variant == "pnp":
            # Same chain cut at every blind (constant-velocity) link: the honestly-tracked pieces.
            by_row = {f["row"]: f for f in frames}
            piece_of, pid = {rows[0]: 0}, 0
            blind = []
            for k, e in enumerate(edges):
                if e[5] == "const_vel":
                    pid += 1
                    blind.append(round(by_row[rows[k]]["t"] - frames[0]["t"], 1))
                piece_of[rows[k + 1]] = pid
            ev = evaluate_method(a.capture, rows, poses, piece_of, metric=True,
                                 extra={"method": name + "_cut", "links": stats, "blind_link_t_s": blind})
            (OUT / f"{a.capture}_{name}_cut_eval.json").write_text(json.dumps(ev, indent=1))
            save_poses(CACHE / "poses" / f"{a.capture}_{name}_cut.json", poses, piece_of, method=name + "_cut")
            print(name + "_cut", {k: ev.get(k) for k in ["n_pieces", "coverage_largest_pct", "longest_run_s"]},
                  "ATE", round(ev["largest"]["ate_rmse_m"], 3), "blind at", blind, flush=True)


if __name__ == "__main__":
    main()

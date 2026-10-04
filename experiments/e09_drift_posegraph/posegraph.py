"""Drift correction: submap pose graph with ICP loop closures (Open3D).

Submaps = ~2.5 s windows anchored at their first frame. Odometry edges = ARKit relative pose
between consecutive anchors (VIO is accurate short-term). Loop edges = point-to-plane ICP between
non-consecutive submaps (> 10 s apart, anchors within 2.5 m) initialised from ARKit; accepted if
fitness > 0.35, inlier RMSE < 2 cm and the correction is < 0.25 m / 5 deg (rejects aliasing).
Corrected frame pose = T_anchor_new @ inv(T_anchor_old) @ T_frame_old.
Output: cache/e09/<capture>_T_wc_corrected.npy (N,4,4) + out/<capture>_posegraph.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np, open3d as o3d
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e09_drift_posegraph/out"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache/e09"; CACHE.mkdir(parents=True, exist_ok=True)
reg = o3d.pipelines.registration
WIN_S, FRAME_STEP, VOX = 2.5, 6, 0.03

def build_submaps(cap, T):
    t = cap.timestamps; subs = []; start = 0
    while start < len(cap) - 10:
        end = int(np.searchsorted(t, t[start] + WIN_S))
        rows = list(range(start, min(end, len(cap)), FRAME_STEP))
        Ta = T[start]; Tinv = np.linalg.inv(Ta)
        pts = np.concatenate([cap.points_world(i, min_conf=2, max_depth=3.5, stride=2) for i in rows])
        pts = pts @ Tinv[:3, :3].T + Tinv[:3, 3]  # into anchor frame
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(VOX)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * VOX, max_nn=30))
        subs.append({"anchor": start, "end": min(end, len(cap)), "pc": pc})
        start = end
    return subs

def run(name, verbose=True):
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    t0 = time.time(); subs = build_submaps(cap, T)
    A = np.stack([T[s["anchor"]] for s in subs])
    pg = reg.PoseGraph()
    for a in A: pg.nodes.append(reg.PoseGraphNode(a))
    info_odo = np.eye(6) * 1e4
    for k in range(len(subs) - 1):
        rel = np.linalg.inv(A[k + 1]) @ A[k]   # transforms points in k frame into k+1 frame
        pg.edges.append(reg.PoseGraphEdge(k, k + 1, rel, info_odo, uncertain=False))
    loops, tried = [], 0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            if t[subs[j]["anchor"]] - t[subs[i]["anchor"]] < 10: continue
            if np.linalg.norm(A[i][:3, 3] - A[j][:3, 3]) > 2.5: continue
            tried += 1
            init = np.linalg.inv(A[j]) @ A[i]
            r = reg.registration_icp(subs[i]["pc"], subs[j]["pc"], 0.06, init, reg.TransformationEstimationPointToPlane())
            r = reg.registration_icp(subs[i]["pc"], subs[j]["pc"], 0.025, r.transformation, reg.TransformationEstimationPointToPlane())
            d = np.linalg.inv(init) @ r.transformation
            dt = np.linalg.norm(d[:3, 3]); dr = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
            if r.fitness > 0.35 and r.inlier_rmse < 0.02 and dt < 0.25 and dr < 5:
                info = reg.get_information_matrix_from_point_clouds(subs[i]["pc"], subs[j]["pc"], 0.025, r.transformation)
                pg.edges.append(reg.PoseGraphEdge(i, j, r.transformation, info, uncertain=True))
                loops.append({"i": i, "j": j, "fitness": round(r.fitness, 3), "rmse_mm": round(1000 * r.inlier_rmse, 1), "corr_mm": round(1000 * dt, 1), "corr_deg": round(dr, 2)})
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.025, edge_prune_threshold=0.25, preference_loop_closure=1.0, reference_node=0)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(), opt)
    A_new = np.stack([n.pose for n in pg.nodes])
    Tc = T.copy()
    for k, s in enumerate(subs):
        Tc[s["anchor"]:s["end"]] = A_new[k] @ np.linalg.inv(A[k]) @ T[s["anchor"]:s["end"]]
    np.save(CACHE / f"{name}_T_wc_corrected.npy", Tc)
    kept = [e for e in pg.edges if e.uncertain]
    shift = np.linalg.norm(Tc[:, :3, 3] - T[:, :3, 3], axis=1)
    summ = {"capture": name, "n_submaps": len(subs), "loop_candidates": tried, "loops_accepted": len(loops), "loops_after_prune": len(kept),
            "loop_corr_mm_median": float(np.median([l["corr_mm"] for l in loops])) if loops else None,
            "pose_shift_mm_p50_p95_max": (1000 * np.percentile(shift, [50, 95, 100])).round(1).tolist(), "sec": round(time.time() - t0, 1)}
    (OUT / f"{name}_posegraph.json").write_text(json.dumps({**summ, "loops": loops}, indent=1))
    if verbose: print(json.dumps(summ))
    return Tc

if __name__ == "__main__":
    for name in sys.argv[1:] or ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
        run(name)

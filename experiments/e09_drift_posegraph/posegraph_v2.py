"""Pose graph v2: fixes for the v1 failure modes seen in ablation.json.

v1 made floor_only worse (revisit median 9.3 -> 21.3 mm) and fattened tails for with_ceiling.
Changes, each targeting a suspected cause:
  1. degenerate loops: ICP along a corridor slides; reject loops whose translational information
     has min/max eigenvalue ratio < 0.05 (unconstrained direction), and require fitness > 0.5.
  2. weighting: odometry edges get the point-cloud information matrix of the consecutive overlap
     (same units as loop edges) times ODO_TRUST, instead of an arbitrary 1e4 * I.
  3. gravity: ARKit roll/pitch is reliable; after optimisation each anchor's rotation correction is
     projected onto a yaw about world +y.
"""
import json, sys, time
from pathlib import Path
import numpy as np, open3d as o3d
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).parent))
from posegraph import build_submaps, ROOT, CACHE, OUT, reg
from roomscan.stray import StrayCapture

ODO_TRUST = 4.0

def yaw_only(R):
    """Closest rotation about world y to R (keeps heading change, drops roll/pitch change)."""
    f = R @ np.array([0, 0, 1.0]); yaw = np.arctan2(f[0], f[2])
    return Rotation.from_euler("y", yaw).as_matrix()

def run(name):
    cap = StrayCapture.load(ROOT / name); T = cap.T_wc(); t = cap.timestamps
    t0 = time.time(); subs = build_submaps(cap, T)
    A = np.stack([T[s["anchor"]] for s in subs]); pg = reg.PoseGraph()
    for a in A: pg.nodes.append(reg.PoseGraphNode(a))
    for k in range(len(subs) - 1):
        rel = np.linalg.inv(A[k + 1]) @ A[k]
        info = reg.get_information_matrix_from_point_clouds(subs[k]["pc"], subs[k + 1]["pc"], 0.03, rel)
        if np.trace(info) < 1e3: info = np.eye(6) * 1e4   # no overlap between consecutive submaps: trust VIO
        pg.edges.append(reg.PoseGraphEdge(k, k + 1, rel, info * ODO_TRUST, uncertain=False))
    loops, rej = [], {"fitness": 0, "rmse": 0, "big": 0, "degenerate": 0}; tried = 0
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            if t[subs[j]["anchor"]] - t[subs[i]["anchor"]] < 10 or np.linalg.norm(A[i][:3, 3] - A[j][:3, 3]) > 2.5: continue
            tried += 1; init = np.linalg.inv(A[j]) @ A[i]
            r = reg.registration_icp(subs[i]["pc"], subs[j]["pc"], 0.06, init, reg.TransformationEstimationPointToPlane())
            r = reg.registration_icp(subs[i]["pc"], subs[j]["pc"], 0.025, r.transformation, reg.TransformationEstimationPointToPlane())
            d = np.linalg.inv(init) @ r.transformation; dt = np.linalg.norm(d[:3, 3])
            dr = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
            if r.fitness < 0.5: rej["fitness"] += 1; continue
            if r.inlier_rmse > 0.02: rej["rmse"] += 1; continue
            if dt > 0.25 or dr > 5: rej["big"] += 1; continue
            info = reg.get_information_matrix_from_point_clouds(subs[i]["pc"], subs[j]["pc"], 0.025, r.transformation)
            ev = np.linalg.eigvalsh(info[3:, 3:])
            if ev[0] / ev[-1] < 0.05: rej["degenerate"] += 1; continue
            pg.edges.append(reg.PoseGraphEdge(i, j, r.transformation, info, uncertain=True))
            loops.append({"i": i, "j": j, "fitness": round(r.fitness, 3), "corr_mm": round(1000 * dt, 1), "eig_ratio": round(float(ev[0] / ev[-1]), 3)})
    opt = reg.GlobalOptimizationOption(max_correspondence_distance=0.025, edge_prune_threshold=0.25, preference_loop_closure=1.0, reference_node=0)
    reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(), reg.GlobalOptimizationConvergenceCriteria(), opt)
    A_new = np.stack([n.pose for n in pg.nodes])
    Tc = T.copy()
    for k, s in enumerate(subs):
        D = A_new[k] @ np.linalg.inv(A[k])                     # world-frame correction of this submap
        D[:3, :3] = yaw_only(D[:3, :3])
        D[:3, 3] = A_new[k][:3, 3] - D[:3, :3] @ A[k][:3, 3]   # keep the optimised anchor position
        Tc[s["anchor"]:s["end"]] = D @ T[s["anchor"]:s["end"]]
    np.save(CACHE / f"{name}_T_wc_corrected_v2.npy", Tc)
    shift = np.linalg.norm(Tc[:, :3, 3] - T[:, :3, 3], axis=1)
    summ = {"capture": name, "n_submaps": len(subs), "loop_candidates": tried, "loops_accepted": len(loops), "rejected": rej,
            "loops_after_prune": sum(e.uncertain for e in pg.edges),
            "pose_shift_mm_p50_p95_max": (1000 * np.percentile(shift, [50, 95, 100])).round(1).tolist(), "sec": round(time.time() - t0, 1)}
    (OUT / f"{name}_posegraph_v2.json").write_text(json.dumps({**summ, "loops": loops}, indent=1)); print(json.dumps(summ))

if __name__ == "__main__":
    for name in sys.argv[1:] or ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
        run(name)

"""Cross-capture consistency: register captures of the same apartment and map disagreement.

Both captures share a gravity-aligned world (y up), so alignment is yaw + 3D translation.
Global init: FPFH + RANSAC on 5 cm voxels; refine with point-to-plane ICP. The residual map
(on mid-height wall points) shows where the two captures' geometry disagrees: a rigid
transform cannot absorb drift, so drift appears as spatially structured residual.
"""
import json, itertools
from pathlib import Path
import numpy as np, open3d as o3d, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]; FUS = ROOT / "experiments/e02_fusion/out"; OUT = ROOT / "experiments/e03_drift/out"
reg = o3d.pipelines.registration

def prep(name, v=0.05):
    pc = o3d.io.read_point_cloud(str(FUS / f"{name}_vox2cm.ply"))
    P = np.asarray(pc.points); floor = np.percentile(P[:, 1], 2)
    pc = pc.select_by_index(np.nonzero((P[:, 1] > floor + 0.3) & (P[:, 1] < floor + 2.0))[0])  # walls/furniture, no floor/ceiling
    d = pc.voxel_down_sample(v); d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2 * v, max_nn=30))
    f = reg.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=5 * v, max_nn=100))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30))
    return pc, d, f

names = ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]
data = {n: prep(n) for n in names}
res = {}
for a, b in itertools.combinations(names, 2):
    (pa, da, fa), (pb, db, fb) = data[a], data[b]
    best = None
    for seed in range(3):
        o3d.utility.random.seed(seed)
        r = reg.registration_ransac_based_on_feature_matching(da, db, fa, fb, True, 0.1,
              reg.TransformationEstimationPointToPoint(False), 3,
              [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.1)],
              reg.RANSACConvergenceCriteria(400000, 0.999))
        if best is None or r.fitness > best.fitness: best = r
    icp = reg.registration_icp(pa, pb, 0.05, best.transformation, reg.TransformationEstimationPointToPlane())
    icp = reg.registration_icp(pa, pb, 0.02, icp.transformation, reg.TransformationEstimationPointToPlane())
    T = icp.transformation
    pa_t = o3d.geometry.PointCloud(pa); pa_t.transform(T)
    dist = np.asarray(pa_t.compute_point_cloud_distance(pb))
    ov = dist < 0.15
    R = T[:3, :3]
    res[f"{a}->{b}"] = {"ransac_fitness": round(best.fitness, 3), "icp_fitness_2cm": round(icp.fitness, 3),
                        "icp_rmse_mm": round(1000 * icp.inlier_rmse, 1),
                        "tilt_deg": round(float(np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))), 3),
                        "overlap_frac_15cm": round(float(ov.mean()), 3),
                        "dist_in_overlap_mm": {"median": round(1000 * float(np.median(dist[ov])), 1),
                                               "p90": round(1000 * float(np.percentile(dist[ov], 90)), 1)}}
    print(a, "->", b, json.dumps(res[f"{a}->{b}"]))
    Q = np.asarray(pa_t.points)
    fig, ax = plt.subplots(figsize=(9, 9))
    Pb = np.asarray(pb.points); ax.scatter(Pb[::4, 0], Pb[::4, 2], s=0.2, c="lightgray")
    sc = ax.scatter(Q[ov, 0], Q[ov, 2], s=0.3, c=1000 * dist[ov], cmap="inferno_r", vmin=0, vmax=60)
    plt.colorbar(sc, label="distance to other capture (mm)"); ax.set_aspect("equal"); ax.set_title(f"{a} aligned onto {b}")
    fig.savefig(OUT / f"cross_{a}__{b}.png", dpi=90); plt.close(fig)
(OUT / "cross_capture.json").write_text(json.dumps(res, indent=1))

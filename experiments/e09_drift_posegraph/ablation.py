"""Drift correction ON vs OFF, measured without ground truth.

1. Revisit residual (e03 method) using original vs corrected poses. Partly circular: loop
   closures were fitted on the same depth, so improvement here is expected, not proof.
2. Cross-capture agreement: floor_only and with_ceiling are corrected INDEPENDENTLY, fused,
   registered rigidly; distance distribution on overlapping wall-band points. This is the
   independent check: two drift-free reconstructions of one apartment should coincide.
3. Wall sharpness: median |point-to-plane| spread of mid-height points around local planes.
"""
import json, sys
from pathlib import Path
import numpy as np, open3d as o3d
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e03_drift"))
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e09_drift_posegraph/out"; CACHE = ROOT / "cache/e09"
reg = o3d.pipelines.registration

def residual(cap, i, j, Ts):
    d = cap.depth_m(i); c = cap.confidence(i); K = cap.K_depth(i)
    v, u = np.nonzero((c == 2) & (d < 4.0)); z = d[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    pw = pc @ Ts[i, :3, :3].T + Ts[i, :3, 3]; pj = (pw - Ts[j, :3, 3]) @ Ts[j, :3, :3]
    Kj = cap.K_depth(j); pj = pj[pj[:, 2] > 0.1]
    uj = np.round(Kj[0, 0] * pj[:, 0] / pj[:, 2] + Kj[0, 2]).astype(int); vj = np.round(Kj[1, 1] * pj[:, 1] / pj[:, 2] + Kj[1, 2]).astype(int)
    ok = (uj >= 0) & (uj < 256) & (vj >= 0) & (vj < 192)
    if ok.sum() < 2000: return None
    ok2 = cap.confidence(j)[vj[ok], uj[ok]] == 2
    r = cap.depth_m(j)[vj[ok], uj[ok]][ok2] - pj[ok][ok2, 2]; r = r[np.abs(r) < 0.3]
    return float(np.median(np.abs(r))) if len(r) > 1000 else None

def revisit(cap, T0, T1):
    t = cap.timestamps; pos = T0[:, :3, 3]; fwd = T0[:, :3, 2]; idx = np.arange(0, len(cap), 6); pairs = []
    for a in idx:
        b = idx[idx > a]; b = b[t[b] - t[a] > 30]
        ok = (np.linalg.norm(pos[b] - pos[a], axis=1) < 0.4) & (fwd[b] @ fwd[a] > np.cos(np.radians(20)))
        pairs += [(a, j) for j in b[ok]]
    rng = np.random.default_rng(0)
    if len(pairs) > 300: pairs = [pairs[k] for k in rng.choice(len(pairs), 300, replace=False)]
    out = {}
    for lab, T in [("off", T0), ("on", T1)]:
        v = np.array([x for x in (residual(cap, i, j, T) for i, j in pairs) if x is not None])
        if len(v) == 0: out[lab] = {"n": 0}; continue
        out[lab] = {"n": len(v), "median_mm": round(1000 * float(np.median(v)), 1), "p90_mm": round(1000 * float(np.percentile(v, 90)), 1)}
    return out

def fuse(cap, T, step=5):
    pts = []
    for i in range(0, len(cap), step):
        d = cap.depth_m(i)[::2, ::2]; c = cap.confidence(i)[::2, ::2]; K = cap.K_depth(i); K[:2] /= 2
        v, u = np.nonzero((c == 2) & (d < 4.0)); z = d[v, u]
        pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
        pts.append(pc @ T[i, :3, :3].T + T[i, :3, 3])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts))).voxel_down_sample(0.02)
    P = np.asarray(pc.points); f0 = np.percentile(P[:, 1], 2)
    pc = pc.select_by_index(np.nonzero((P[:, 1] > f0 + 0.3) & (P[:, 1] < f0 + 2.0))[0])
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.06, max_nn=30)); return pc

def sharpness(pc):
    """Median distance of points to the plane fitted through their 20 nearest neighbours (wall thickness proxy)."""
    P = np.asarray(pc.points); N = np.asarray(pc.normals); tree = o3d.geometry.KDTreeFlann(pc)
    rng = np.random.default_rng(0); idx = rng.choice(len(P), 20000, replace=False); d = []
    for i in idx:
        if abs(N[i, 1]) > 0.3: continue
        _, nn, _ = tree.search_radius_vector_3d(P[i], 0.08)
        Q = P[np.asarray(nn)]
        if len(Q) < 15: continue
        Q = Q - Q.mean(0); s = np.linalg.svd(Q, compute_uv=False); d.append(s[-1] / np.sqrt(len(Q)))
    return round(1000 * float(np.median(d)), 2)

def cross(pa, pb):
    fpfh = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
    da, db = pa.voxel_down_sample(0.05), pb.voxel_down_sample(0.05)
    for d in (da, db): d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.utility.random.seed(0)
    r = reg.registration_ransac_based_on_feature_matching(da, db, fpfh(da), fpfh(db), True, 0.1, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.1)], reg.RANSACConvergenceCriteria(400000, 0.999))
    T = r.transformation
    for th in (0.05, 0.02): T = reg.registration_icp(pa, pb, th, T, reg.TransformationEstimationPointToPlane()).transformation
    q = o3d.geometry.PointCloud(pa); q.transform(T); dist = np.asarray(q.compute_point_cloud_distance(pb)); ov = dist < 0.15
    return {"overlap": round(float(ov.mean()), 3), "median_mm": round(1000 * float(np.median(dist[ov])), 1), "p90_mm": round(1000 * float(np.percentile(dist[ov], 90)), 1)}

if __name__ == "__main__":
    VER = sys.argv[1] if len(sys.argv) > 1 else ""   # "" = v1, "_v2" = v2 corrected poses
    res = {}; clouds = {}
    for name in ["single_room", "single_scan_floor_only", "single_scan_with_ceiling"]:
        cap = StrayCapture.load(ROOT / name); T0 = cap.T_wc(); T1 = np.load(CACHE / f"{name}_T_wc_corrected{VER}.npy")
        res[name] = {"revisit_gap_gt30s": revisit(cap, T0, T1)}
        clouds[name] = {k: fuse(cap, T) for k, T in [("off", T0), ("on", T1)]}
        res[name]["wall_thickness_mm"] = {k: sharpness(clouds[name][k]) for k in ("off", "on")}
        print(name, json.dumps(res[name]))
    for a, b in [("single_scan_floor_only", "single_scan_with_ceiling"), ("single_room", "single_scan_with_ceiling")]:
        res[f"cross_{a}__{b}"] = {k: cross(clouds[a][k], clouds[b][k]) for k in ("off", "on")}
        print(a, b, json.dumps(res[f"cross_{a}__{b}"]))
    (OUT / f"ablation{VER}.json").write_text(json.dumps(res, indent=1))

"""Is inter-capture disagreement rigid per room? (motivates room-local frames + plane-anchored stitching)

floor_only is globally registered onto with_ceiling (FPFH+RANSAC+ICP). Then, per segmented room
(with_ceiling e10 v2b labels, dilated 0.4 m to include its walls), ICP again on the room crop only.
If residual drops to the noise floor per room, drift is ~rigid within a room and accumulates
between rooms -> room-local measurement is drift-safe and only the stitch needs correction.
"""
import json, sys
from pathlib import Path
import numpy as np, open3d as o3d
from scipy import ndimage as ndi
sys.path.insert(0, str(Path(__file__).parent))
from ablation import fuse, cross, reg
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e09_drift_posegraph/out"
A, B = "single_scan_floor_only", "single_scan_with_ceiling"
ca, cb = StrayCapture.load(ROOT / A), StrayCapture.load(ROOT / B)
pa, pb = fuse(ca, ca.T_wc()), fuse(cb, cb.T_wc())
fpfh = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=100))
da, db = pa.voxel_down_sample(0.05), pb.voxel_down_sample(0.05)
for d in (da, db): d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.utility.random.seed(0)
r = reg.registration_ransac_based_on_feature_matching(da, db, fpfh(da), fpfh(db), True, 0.1, reg.TransformationEstimationPointToPoint(False), 3,
    [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(0.1)], reg.RANSACConvergenceCriteria(400000, 0.999))
Tg = r.transformation
for th in (0.05, 0.02): Tg = reg.registration_icp(pa, pb, th, Tg, reg.TransformationEstimationPointToPlane()).transformation
pa.transform(Tg)
grid = json.loads((ROOT / f"experiments/e10_room_seg/out/{B}_v2_grid.json").read_text())
lab = np.load(ROOT / f"experiments/e10_room_seg/out/{B}_v2b_h1.9_labels.npy")
def room_of(P):
    ij = np.floor((P[:, [0, 2]] - np.array(grid["origin_xz"])) / grid["res"]).astype(int)
    ij = np.clip(ij, 0, np.array(lab.shape) - 1); return ij
def stats(d): ov = d < 0.15; return {"median_mm": round(1000 * float(np.median(d[ov])), 1), "p90_mm": round(1000 * float(np.percentile(d[ov], 90)), 1), "overlap": round(float(ov.mean()), 3)}
res = {}
for k in range(1, lab.max() + 1):
    m = ndi.binary_dilation(lab == k, iterations=int(0.4 / grid["res"]))
    ia, ib = room_of(np.asarray(pa.points)), room_of(np.asarray(pb.points))
    qa = pa.select_by_index(np.nonzero(m[ia[:, 0], ia[:, 1]])[0]); qb = pb.select_by_index(np.nonzero(m[ib[:, 0], ib[:, 1]])[0])
    if len(qa.points) < 2000 or len(qb.points) < 2000: continue
    before = np.asarray(qa.compute_point_cloud_distance(qb))
    T = reg.registration_icp(qa, qb, 0.05, np.eye(4), reg.TransformationEstimationPointToPlane()).transformation
    T = reg.registration_icp(qa, qb, 0.02, T, reg.TransformationEstimationPointToPlane()).transformation
    q = o3d.geometry.PointCloud(qa); q.transform(T); after = np.asarray(q.compute_point_cloud_distance(qb))
    yaw = np.degrees(np.arctan2(T[0, 2], T[2, 2]))
    res[f"room{k}"] = {"global_only": stats(before), "per_room_rigid": stats(after), "local_shift_mm": round(1000 * float(np.linalg.norm(T[:3, 3])), 1), "local_yaw_deg": round(float(yaw), 2)}
    print(k, json.dumps(res[f"room{k}"]))
(OUT / "per_room_rigid.json").write_text(json.dumps(res, indent=1))

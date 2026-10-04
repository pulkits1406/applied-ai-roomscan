"""Run MoGe-2 (FOV given, as from EXIF) on every photo candidate; cache per-frame results.

Per frame (pseudo-GT from LiDAR / ARKit):
  scale_err   = median(pred / lidar) - 1 on confidence-2 pixels  (>0: model overestimates)
  absrel      = AbsRel after per-frame median scaling (shape quality)
  cam_h_pred  = camera height above the floor plane fitted to MoGe points (RANSAC on the lower
                image region, plane normal within 20 deg of the image-up axis); None if no floor
  cam_h_true  = camera y - floor y (floor from fused LiDAR, per-room median)
Raw predicted depth (256x192, landscape) cached to cache/e12/<row>.npy.
"""
import json
from pathlib import Path
import numpy as np, cv2, torch
from moge.model.v2 import MoGeModel
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments/e12_photo_scale/out"; CACHE = ROOT / "cache/e12"; CACHE.mkdir(parents=True, exist_ok=True)
sets = json.loads((OUT / "photo_sets.json").read_text())
CAPN = sets["capture"]; cap = StrayCapture.load(ROOT / CAPN)
meta = {m["row"]: m for m in json.loads((ROOT / f"cache/frames/{CAPN}/frames.json").read_text())["frames"]}
grid = json.loads((ROOT / f"experiments/e10_room_seg/out/{CAPN}_v2_grid.json").read_text())
dev = torch.device("mps"); model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(dev).eval()
rng = np.random.default_rng(0)

def floor_height(pts_up):
    """pts_up: (N,3) camera-frame points of the upright image (x right, y down, z fwd)."""
    best = None
    for _ in range(300):
        s = pts_up[rng.choice(len(pts_up), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9: continue
        n /= nn
        if n[1] < 0: n = -n
        if n[1] < np.cos(np.radians(20)): continue          # floor normal ~ image down axis (+y)
        d = -n @ s[0]; inl = np.abs(pts_up @ n + d) < 0.02
        if best is None or inl.sum() > best[0]: best = (inl.sum(), n, d)
    if best is None or best[0] < 0.05 * len(pts_up): return None
    return float(abs(best[2]))

res = {}
for room, rows in sets["rooms"].items():
    for r in rows:
        bgr = cv2.imread(str(ROOT / f"cache/frames/{CAPN}/rgb/{r:06d}.jpg"))
        up = cv2.cvtColor(cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)
        K = np.array(meta[r]["K"]); fov_x = float(np.degrees(2 * np.arctan(720 / (2 * K[1, 1]))))  # upright width = landscape height
        x = torch.tensor(up / 255.0, dtype=torch.float32, device=dev).permute(2, 0, 1)
        with torch.no_grad(): o = model.infer(x, fov_x=fov_x)
        pts = o["points"].float().cpu().numpy(); msk = o["mask"].cpu().numpy() > 0.5
        dep = o["depth"].float().cpu().numpy(); dep[~msk] = np.nan
        lower = pts[int(0.55 * pts.shape[0]):][msk[int(0.55 * pts.shape[0]):]]
        lower = lower[np.isfinite(lower).all(1)]
        ch = floor_height(lower[rng.choice(len(lower), min(20000, len(lower)), replace=False)]) if len(lower) > 500 else None
        d_land = cv2.resize(cv2.rotate(dep, cv2.ROTATE_90_COUNTERCLOCKWISE), (256, 192), interpolation=cv2.INTER_AREA)
        np.save(CACHE / f"{r:06d}.npy", d_land.astype(np.float16))
        lid = cap.depth_m(r); cf = cap.confidence(r)
        v = (cf == 2) & np.isfinite(d_land) & (lid < 6)
        ratio = d_land[v] / lid[v]; s = float(np.median(ratio))
        absrel = float(np.median(np.abs(d_land[v] / s - lid[v]) / lid[v]))
        T = np.array(meta[r]["T_wc"]); i, j = np.floor((T[[0, 2], 3] - np.array(grid["origin_xz"])) / grid["res"]).astype(int)
        res[r] = {"room": room, "scale_err": s - 1, "absrel": absrel, "cam_h_pred": ch, "cam_y": float(T[1, 3]), "fov_x": fov_x}
    print("room", room, "done", len(rows))
# per-room floor height from fused LiDAR cloud near each room's cameras
import open3d as o3d
P = np.asarray(o3d.io.read_point_cloud(str(ROOT / f"experiments/e02_fusion/out/{CAPN}_vox2cm.ply")).points)
f0 = grid["floor_y"]
for room, rows in sets["rooms"].items():
    c = np.array([np.array(meta[r]["T_wc"])[[0, 2], 3] for r in rows]).mean(0)
    near = (np.linalg.norm(P[:, [0, 2]] - c, axis=1) < 1.5) & (np.abs(P[:, 1] - f0) < 0.06)
    fy = float(np.median(P[near, 1])) if near.sum() > 100 else f0
    for r in rows: res[r]["cam_h_true"] = res[r]["cam_y"] - fy; res[r]["room_floor_y"] = fy
(OUT / "moge_per_frame.json").write_text(json.dumps(res, indent=1))
print("frames", len(res))

"""Doorway centres between segmented rooms (e10 v2b labels), for SIMULATING the protocol's
doorway photos. Evaluation-side only: used to choose which walkthrough frames stand in for
"a photo taken standing in the doorway"; the photo algorithm never receives these positions.
"""
import json
from pathlib import Path
import numpy as np, open3d as o3d
from scipy import ndimage as ndi
from roomscan import geometry as G

ROOT = Path(__file__).resolve().parents[2]; CAP = "single_scan_with_ceiling"
pc = o3d.io.read_point_cloud(str(ROOT / f"experiments/e02_fusion/out/{CAP}_vox2cm.ply"))
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
seg = G.segment_rooms(np.asarray(pc.points), np.asarray(pc.normals))
ref = np.load(ROOT / f"experiments/e10_room_seg/out/{CAP}_v2b_h1.9_labels.npy")
same = ref.shape == seg.labels.shape and bool((ref == seg.labels).all())
centers = seg.grid.centers(); out = []
dlab, nd = ndi.label(seg.doorway)
for k in range(1, nd + 1):
    cells = dlab == k
    ring = ndi.binary_dilation(cells, iterations=3) & ~cells
    touching = sorted({int(x) for x in np.unique(seg.labels[ring]) if x > 0})
    if len(touching) != 2 or cells.sum() < 3: continue
    pts = centers[cells]; ev, evec = np.linalg.eigh(np.cov(pts.T)); width = float(np.ptp(pts @ evec[:, -1]) + G.GRID)
    if 0.5 <= width <= 2.5:
        out.append({"rooms": touching, "center_xz": pts.mean(0).round(3).tolist(), "width_m_coarse": round(width, 2), "axis_xz": evec[:, -1].round(3).tolist()})
res = {"capture": CAP, "labels_identical_to_e10_v2b": same, "doorways": out}
(Path(__file__).parent / "doorways.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

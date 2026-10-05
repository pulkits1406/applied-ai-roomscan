"""Room outline from single photos (no cross-photo registration) [PSEUDO-GT = LiDAR v1.1 rooms].

Per photo: MoGe points x room ensemble scale (MoGe -> metric = sqrt(C_MA * r) / r), gravity from
that photo's own camera axes + normals, then the shared cloud -> room step (cloudroom). Reports
each photo's outline: number of walls, measured/inferred faces, axis-aligned extents, and the LiDAR
room's extents for comparison (rectangular rooms only are directly comparable)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from roomscan import geometry as G
from roomscan.cloudroom import gravity_rotation, point_normals, room_from_cloud, voxel
from roomscan.photo import inference, reconstruct as RC
from roomscan.photo.images import load_photo, room_folders

ROOT = Path(__file__).resolve().parents[2]


def lidar_rooms(plan="runs/e22/ladder/plan.json"):
    p = json.loads((ROOT / plan).read_text())
    S = {s["id"]: s for s in p["surfaces"]}
    return {r["id"]: {"n_walls": len(r["wall_ids"]), "area": round(r["floor_area"]["value"], 2),
                      "lengths": sorted(round(S[w]["length"]["value"], 3) for w in r["wall_ids"]),
                      "ceiling": r["ceiling_height"]["value"]} for r in p["rooms"]}


def main(folder="cache/photo_sim/single_scan_with_ceiling_K6", cache="runs/e23/cache"):
    lid = lidar_rooms()
    rep = {}
    for name, paths in room_folders(ROOT / folder).items():
        photos = [load_photo(p) for p in paths]
        mo = inference.moge(photos, ROOT / cache / name)
        ma = inference.mapanything(photos, ROOT / cache / name)
        s_moge = RC.room_cloud(mo, ma, "moge").scale_ma_to_m / RC.room_cloud(mo, ma, "moge").r_moge_over_ma
        rows = []
        for i, m in enumerate(mo):
            p, n = point_normals((m["points"] * s_moge)[::2, ::2], m["mask"][::2, ::2])
            Rg = gravity_rotation(np.eye(3)[None], n)
            P, N = voxel(p @ Rg.T, n @ Rg.T)
            g = room_from_cloud(P, N, np.zeros((1, 3)))
            if isinstance(g, str):
                rows.append({"photo": i, "result": g}); continue
            uv = g.walls.poly_uv
            L = np.linalg.norm(np.roll(uv, -1, 0) - uv, axis=1)
            rows.append({"photo": i, "result": "ok", "step": g.step, "n_walls": len(uv), "sources": g.walls.source,
                         "measured_faces": sum(s in ("close", "all") for s in g.walls.source), "extent_m": np.round(np.ptp(uv, 0), 3).tolist(),
                         "lengths": np.round(L, 3).tolist(), "ceiling_h": None if g.ceil_h is None else round(g.ceil_h, 3),
                         "area": round(abs(G.polygon_area(uv)), 2)})
        rep[name] = {"lidar": lid.get(name), "scale_moge_to_m": round(s_moge, 4), "photos": rows}
        print(name, "LiDAR:", lid.get(name))
        for r in rows:
            print("   ", {k: r[k] for k in ("photo", "result", "step", "n_walls", "measured_faces", "extent_m", "ceiling_h", "area") if k in r})
    (ROOT / "experiments/e23_photo_frontend/out/per_photo.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main(*sys.argv[1:])

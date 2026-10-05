"""Cache the v1 per-room inputs to wall extraction (room clouds, region, other-room cells) so the
small-room failure can be studied without re-fusing. Output: cache/e22/<capture>/room<k>.npz,
seg.npz. Uses the production modules unchanged (roomscan.lidar.rooms)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from roomscan import geometry as G
from roomscan.lidar import rooms as RM
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]


def main(capture="single_scan_with_ceiling", registration="icp"):
    out = ROOT / "cache/e22" / (capture if registration == "icp" else f"{capture}_{registration}")
    out.mkdir(parents=True, exist_ok=True)
    c = RM.prepare(StrayCapture.load(ROOT / capture))
    seg = c.seg
    np.savez_compressed(out / "seg.npz", labels=seg.labels, barrier=seg.barrier, floor=seg.floor, doorway=seg.doorway,
                        origin=seg.grid.origin, floor_y=seg.grid.floor_y, room_of_frame=c.room_of_frame)
    centers = seg.grid.centers()
    for label in range(1, seg.labels.max() + 1):
        rc = RM.room_cloud(c, label, registration=registration)
        if rc is None:
            print(label, "no visits"); continue
        np.savez_compressed(out / f"room{label}.npz", P=np.asarray(rc.all.points), N=np.asarray(rc.all.normals),
                            CP=np.asarray(rc.close.points), CN=np.asarray(rc.close.normals),
                            region=centers[G.fill_region(seg.labels, label)],
                            other=centers[(seg.labels > 0) & (seg.labels != label)], visits=np.array(rc.visits),
                            aligned=np.array([a.get("aligned", False) for a in rc.alignment], bool))
        print(label, len(rc.all.points), len(rc.close.points), "visits", rc.visits, rc.alignment)


if __name__ == "__main__":
    main(*sys.argv[1:])

"""One command per LiDAR capture:  uv run roomscan-lidar <stray_dir> <out_dir> [--placement arkit|arkit_yaw]"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roomscan.lidar.build import build
from roomscan.render import render


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--placement", default="arkit_yaw", choices=["arkit", "arkit_yaw"])
    ap.add_argument("--no-ladder", action="store_true", help="reproduce v1.0: drop rooms whose topology fails instead of escalating (e22)")
    ap.add_argument("--seg-pad", nargs=2, type=float, metavar=("LOW", "HIGH"), default=None,
                    help="segmentation grid margin in m (default 1.0 1.0); 0.3 0.0 reproduces the pre-fix behaviour (reports/fix_loop)")
    a = ap.parse_args()
    t0 = time.time()
    kw = {"seg_pad": tuple(a.seg_pad)} if a.seg_pad else {}
    plan, dbg = build(a.capture, a.placement, ladder=not a.no_ladder, **kw)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=1))
    render(plan, out / "plan.png", title=f"LiDAR plan: {Path(a.capture).name}")
    dbg["seconds"] = round(time.time() - t0, 1)
    (out / "debug.json").write_text(json.dumps(dbg, indent=1, default=str))
    print(json.dumps({k: v for k, v in dbg.items() if k != "rooms"}, indent=1, default=str))


if __name__ == "__main__":
    main()

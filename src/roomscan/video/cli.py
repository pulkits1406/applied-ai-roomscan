"""One command per video capture:  uv run roomscan-video <clip.mov|mp4> <out_dir> [--rotate auto|0|90|180|270]

The video must be a plain handheld clip; nothing but its frames and timestamps is used. Files
without rotation metadata (Stray Scanner rgb.mp4) need --rotate 90."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roomscan.render import render
from roomscan.video.build import build


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--rotate", default="auto", help="auto = container metadata; else degrees clockwise")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--cache", default=None, help="frame/model cache (default <out>/cache)")
    a = ap.parse_args(argv)
    src = Path(a.capture)
    rotate = a.rotate
    if rotate == "auto" and src.name == "rgb.mp4" and (src.parent / "odometry.csv").exists():
        rotate = 90          # Stray Scanner export: sensor-orientation frames without rotation metadata, phone held upright
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan, dbg = build(src, a.cache or out / "cache", rotate, a.fps)
    (out / "plan.json").write_text(plan.model_dump_json(indent=1))
    render(plan, out / "plan.png", title=f"Video plan: {src.name}")
    dbg["seconds"] = round(time.time() - t0, 1)
    (out / "debug.json").write_text(json.dumps(dbg, indent=1, default=str))
    print(json.dumps({k: v for k, v in dbg.items() if k != "windows"}, default=str), *plan.quality_flags, sep="\n")


if __name__ == "__main__":
    main()

"""One command per photo capture:  uv run roomscan-photo <folder_of_room_folders> <out_dir>"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from roomscan.photo.build import build
from roomscan.render import render


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("capture"); ap.add_argument("out")
    ap.add_argument("--mode", default="moge", choices=["moge", "ma"], help="per-photo geometry source (e23)")
    ap.add_argument("--cache", default=None, help="model output cache (default <out>/cache)")
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    plan, dbg = build(a.capture, a.cache or out / "cache", a.mode)
    (out / "plan.json").write_text(plan.model_dump_json(indent=1))
    render(plan, out / "plan.png", title=f"Photo plan: {Path(a.capture).name}")
    dbg["seconds"] = round(time.time() - t0, 1)
    (out / "debug.json").write_text(json.dumps(dbg, indent=1, default=str))
    print(json.dumps({k: v for k, v in dbg.items() if k != "rooms"}, default=str), *plan.quality_flags, sep="\n")


if __name__ == "__main__":
    main()

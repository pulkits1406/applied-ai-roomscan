"""Which real room does each video room come from? (evaluation only: ARKit room labels of the
frames in the window's time span). The evaluator's Hungarian perimeter matching for unlabelled
rooms can pair a room with the wrong reference; this checks identity and scores by identity."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from roomscan.lidar import rooms as RM
from roomscan.stray import StrayCapture

ROOT = Path(__file__).resolve().parents[2]


def main(run="runs/e24/three_tier", capture="single_scan_with_ceiling", lidar_plan="runs/e24/lidar_v11/plan.json"):
    plan = json.loads((ROOT / run / "video_with_ceiling/plan.json").read_text())
    ev = json.loads((ROOT / run / "eval/pseudo_gt_lidar_v11.json").read_text())
    vid = next(c for c in ev["captures"] if c["capture"] == "video_with_ceiling")
    matched = {a["room"]: a["pred"] for a in vid["areas"]}
    lid = {r["id"]: r for r in json.loads((ROOT / lidar_plan).read_text())["rooms"]}
    cap = StrayCapture.load(ROOT / capture)
    c = RM.prepare(cap)
    t = cap.timestamps - cap.timestamps[0]
    rows = []
    spans = {f.split("_time_span")[0]: tuple(map(float, re.search(r"from ([\d.]+)-([\d.]+) s", f).groups()))
             for f in plan["quality_flags"] if "_time_span" in f}
    for r in plan["rooms"]:
        if r.get("label"):                 # plans made before time spans moved to quality_flags
            t0, t1 = map(float, re.search(r"_t(\d+)-(\d+)s", r["label"]).groups())
        else:
            t0, t1 = spans[r["id"]]
        labs = c.room_of_frame[(t >= t0) & (t <= t1)]
        labs = labs[labs > 0]
        truth = f"r{int(np.bincount(labs).argmax())}" if len(labs) else None
        frac = float(np.mean(labs == int(truth[1:]))) if truth else 0.0
        a = r["floor_area"]["value"]
        ref = lid.get(truth)
        rows.append({"room": r["id"], "span_s": [t0, t1], "area": round(a, 2), "truth_room": truth, "truth_frac": round(frac, 2),
                     "truth_area": None if ref is None else round(ref["floor_area"]["value"], 2),
                     "area_rel_err_by_identity": None if ref is None else round(a / ref["floor_area"]["value"] - 1, 3),
                     "evaluator_matched_to": [k for k, v in matched.items() if abs(v - a) < 1e-6]})
        print(rows[-1])
    (ROOT / "experiments/e24_three_tier_eval/out").mkdir(exist_ok=True)
    (ROOT / "experiments/e24_three_tier_eval/out/video_identity.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])

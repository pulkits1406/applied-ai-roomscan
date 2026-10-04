"""Door cue: how much is detection quality vs the cue itself?

oracle  : keep detections whose LiDAR-measured height is door-like (1.95-2.25 m) -> upper bound of
          the cue if door selection were perfect (uses pseudo-GT, NOT deployable).
image   : deployable filter using only image evidence: box aspect (h/w px) in [1.9, 3.4], box bottom
          in the lowest 20 % of the frame, detector score >= 0.45.
Per room, the scale is the median over the room's kept doors (photos taken there).
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
R = json.loads((OUT / "door_cue_per_detection.json").read_text())
R = [x for x in R if x["h_moge"] and x["h_lidar"]]
H_IMG = 960
def est(xs, H):
    e = np.array([(H / x["h_moge"]) * (1 + x["mono_err"]) - 1 for x in xs])
    return e
res = {}
filters = {
    "oracle_lidar_height_1.95_2.25": lambda x: 1.95 <= x["h_lidar"] <= 2.25,
    "image_only": lambda x: 1.9 <= (x["box"][3] - x["box"][1]) / max(x["box"][2] - x["box"][0], 1) <= 3.4 and x["box"][3] > 0.8 * H_IMG and x["score"] >= 0.45,
}
for name, f in filters.items():
    xs = [x for x in R if f(x)]
    hl = np.array([x["h_lidar"] for x in xs])
    r = {"n": len(xs), "rooms": sorted({x["room"] for x in xs}), "lidar_height_p10_p50_p90": np.percentile(hl, [10, 50, 90]).round(3).tolist() if len(xs) else None}
    for H in [2.03, 2.1, round(float(np.median(hl)), 3) if len(xs) else 2.05]:
        e = est(xs, H)
        if len(e):
            room_e = []
            for room in {x["room"] for x in xs}:
                rx = [x for x in xs if x["room"] == room]; room_e.append(float(np.median(est(rx, H))))
            r[f"prior_{H}"] = {"per_photo_median_abs_pct": round(100 * float(np.median(np.abs(e))), 2), "per_photo_within_8pct": round(float((np.abs(e) <= .08).mean()), 3),
                                "per_room_errs_pct": [round(100 * v, 1) for v in room_e]}
    res[name] = r
(OUT / "door_cue_filter.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))

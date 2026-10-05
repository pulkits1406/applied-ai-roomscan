"""Per-tier diagnostics a development site needs beyond dimensional errors, read from the
pipeline's debug.json (pipeline artifact, not part of the output contract)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def summarise(tier: str, debug_path: Path) -> dict:
    if not debug_path.exists():
        return {}
    d = json.loads(debug_path.read_text())
    if tier == "video":
        w = d.get("windows", [])
        ok = [x for x in w if x.get("result") == "ok"]
        hs = [float(np.std(x["cam_heights"])) for x in w if x.get("cam_heights")]
        return {"duration_s": d.get("video", {}).get("duration_s"), "fov_x_deg_estimated": d.get("fov_x_deg"),
                "windows": len(w), "windows_coherent": len(ok), "coherent_frac": round(len(ok) / len(w), 3) if w else None,
                "window_results": {r: sum(1 for x in w if x.get("result") == r) for r in sorted({x.get("result") for x in w})},
                "camera_height_spread_median_m": round(float(np.median(hs)), 3) if hs else None,
                "longest_coherent_run_windows": _longest([x.get("result") == "ok" for x in w])}
    if tier == "photo":
        return {"rooms": {k: {kk: v.get(kk) for kk in ("photos", "result", "camera_height_spread_m", "stability_runs_without_outline",
                                                        "log_scale_disagreement", "scale_ma_to_m")}
                          for k, v in d.get("rooms", {}).items()}}
    if tier == "lidar":
        return {"upper_walls_observed": d.get("high_observed"), "regions": d.get("n_labels"),
                "rooms": {k: (v if isinstance(v, str) else {"step": v.get("step"), "n_walls": v.get("n_walls"),
                                                            "unaligned_visits": v.get("unaligned_visits"),
                                                            "stability_runs_without_outline": v.get("stability", {}).get("runs_without_outline")})
                          for k, v in d.get("rooms", {}).items()}}
    return {}


def _longest(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best

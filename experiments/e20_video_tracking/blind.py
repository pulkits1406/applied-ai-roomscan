"""Where is the video visually blind? LightGlue match counts between neighbouring keyframes vs ARKit motion.

Usage: uv run python experiments/e20_video_tracking/blind.py <capture>
A link (row i -> next cached row) is "blind" when ALIKED+LightGlue returns < 20 matches. Writes
out/<capture>_blind.json: blind links, merged blind intervals, and their ARKit angular speed / depth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE, OUT, load_frames  # noqa: E402


def main():
    cap = sys.argv[1]
    fr = load_frames(cap)
    M = {}
    for p in sorted((CACHE / "match" / cap).glob("chunk_*.npz")):
        z = np.load(p)
        M.update({k: len(z[k]) for k in z.files})
    t0 = fr[0]["t"]
    res = {}
    for off in (1, 2):
        links = []
        for i in range(len(fr) - off):
            a, b = fr[i], fr[i + off]
            n = M.get(f"{a['row']}_{b['row']}")
            if n is None:
                continue
            links.append((a["t"] - t0, n, a["ang_speed_dps"], a["median_depth_m"]))
        L = np.array(links)
        blind = L[L[:, 1] < 20]
        ivs = []
        for t, *_ in blind:
            if ivs and t - ivs[-1][1] < 0.6:
                ivs[-1][1] = t
            else:
                ivs.append([t, t])
        res[f"offset{off}"] = {
            "n_links": int(len(L)), "n_blind": int(len(blind)), "blind_pct": float(100 * len(blind) / len(L)),
            "blind_intervals_s": [[round(a, 1), round(b - a, 1)] for a, b in ivs], "n_intervals": len(ivs),
            "blind_ang_speed_median_dps": float(np.median(blind[:, 2])) if len(blind) else None,
            "blind_depth_median_m": float(np.median(blind[:, 3])) if len(blind) else None,
            "all_ang_speed_median_dps": float(np.median(L[:, 2])), "all_depth_median_m": float(np.median(L[:, 3])),
            "frac_blind_if_fast80_and_depth_lt0p7": float(np.mean(L[(L[:, 2] > 80) & (L[:, 3] < 0.7), 1] < 20))
            if ((L[:, 2] > 80) & (L[:, 3] < 0.7)).any() else None,
            "matches_median": float(np.median(L[:, 1]))}
    (OUT / f"{cap}_blind.json").write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(cap, k, {kk: vv for kk, vv in v.items() if kk != "blind_intervals_s"})


if __name__ == "__main__":
    main()

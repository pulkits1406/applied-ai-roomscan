"""Part A sweep: MapAnything (images + true K) on seeded, time-spread K-photo subsets per room.

One model load per process; rooms/K/subsets loop inside. Every run's per-view prediction is cached to
cache/e16/runs/r<room>_K<K>_s<si>.npz (resumable: cached runs are re-evaluated, not re-inferred) and one
JSON line per run is appended to out/runs.jsonl. Subsets use e13's draw_subsets with e13's seeds, so
room-1 subsets 0..9 are identical to e13's first 10.

  cache/e13/.venv/bin/python experiments/e16_photo_scale_generalization/run_sweep.py --rooms 3 4 6 7 8 --ks 2 3 4 6 8
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from e16lib import ROOT, SETS, evaluate, infer_views, load_model, save_npz, upright_K, upright_rgb, wait_for_memory
from run_subsets import draw_subsets  # e13 (path added by e16lib)

OUT = ROOT / "experiments/e16_photo_scale_generalization/out"
CACHE = ROOT / "cache/e16/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", nargs="+", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4, 6, 8])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-minutes", type=float, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "runs.jsonl"
    done = set()
    if out_path.exists():
        done = {(d["room"], d["K"], d["subset"]) for d in map(json.loads, open(out_path))}
    rooms = SETS["rooms"]
    model, t_start = None, time.time()
    log = open(OUT / "run_log.txt", "a")

    def say(*x):
        msg = " ".join(map(str, x)); print(msg, flush=True); log.write(msg + "\n"); log.flush()

    with open(out_path, "a") as f:
        for room in a.rooms:
            for K in a.ks:
                if len(rooms[room]) < K:
                    say(f"skip room {room} K={K}: only {len(rooms[room])} candidates"); continue
                for si, rows in enumerate(draw_subsets(rooms[room], K, a.n, seed=1000 * int(room) + K)):
                    if (room, K, si) in done:
                        continue
                    npz = CACHE / f"r{room}_K{K}_s{si}.npz"
                    if not npz.exists():
                        if (time.time() - t_start) / 60 > a.max_minutes:
                            say("time budget reached; stopping"); return
                        mem = wait_for_memory(say)
                        if model is None:
                            model = load_model("mps")
                        res = infer_views(model, [upright_rgb(r) for r in rows], [upright_K(r) for r in rows])
                        save_npz(npz, rows, res)
                    else:
                        mem = "cached"
                    z = np.load(npz)
                    rec = {"room": room, "subset": si, "variant": "images+K",
                           **evaluate(list(z["rows"]), z["depth_land"], z["K_pred"], z["poses"], float(z["sec"]))}
                    f.write(json.dumps(rec) + "\n"); f.flush()
                    say(room, K, si, f"err {rec['scale_err_pct']:+.1f}% focal {rec['focal_err_pct']:+.1f}%",
                        f"fcorr {rec['scale_err_pct_focalcorr']:+.1f}% moge {rec['moge_err_pct']:+.1f}%",
                        f"base {rec.get('baseline_ratio_pred_over_gt') or float('nan'):.3f} {rec['sec']:.1f}s [{mem}]")


if __name__ == "__main__":
    main()

"""Are model outputs identical across independent processes? Compares every cached MoGe and
MapAnything output of two uncached video runs of the same clip (different processes, different
model-loading pattern: per-window MoGe loads in run 1, one shared load in run 2)."""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main(a="runs/e25/video_with_ceiling/cache", b="runs/e24/three_tier/video_with_ceiling/cache"):
    rows = []
    for wa in sorted(glob.glob(f"{ROOT / a}/w*")) + [str(ROOT / a / "fov")]:
        wb = wa.replace(a, b)
        rec = {"set": Path(wa).name}
        for kind in ("moge", "ma"):
            fa, fb = glob.glob(f"{wa}/{kind}_*.npz"), glob.glob(f"{wb}/{kind}_*.npz")
            if not fa or not fb:
                continue
            za, zb = np.load(fa[0]), np.load(fb[0])
            rec[kind] = {k: bool(np.array_equal(za[k], zb[k], equal_nan=True)) for k in za.files if not k.startswith("sec")}
        rows.append(rec)
    ident = all(all(v.values()) for r in rows for k, v in r.items() if k != "set")
    out = {"identical_everywhere": ident, "n_sets": len(rows), "sets": rows}
    (ROOT / "experiments/e26_determinism/out/model_runs.json").write_text(json.dumps(out, indent=1))
    print("sets compared:", len(rows), "bitwise identical:", ident)


if __name__ == "__main__":
    main(*sys.argv[1:])

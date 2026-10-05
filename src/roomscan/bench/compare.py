"""Material comparison of two plans of the same input: identical structure (rooms, walls, openings,
statuses, flags) and the largest numeric difference. Plans of the same input are not bitwise
identical (Open3D's multithreaded reductions change summation order: ~1e-12 m, e26), so
repeatability is judged with a tolerance, never by bytes."""
from __future__ import annotations

import json
import sys

from roomscan.model import Plan


def _flatten(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            if k != "provenance":
                yield from _flatten(v, f"{prefix}.{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def compare(a: Plan, b: Plan) -> dict:
    fa, fb = dict(_flatten(a.model_dump())), dict(_flatten(b.model_dump()))
    keys_differ = sorted(set(fa) ^ set(fb))
    nonnum = [k for k in set(fa) & set(fb) if not isinstance(fa[k], (int, float)) or isinstance(fa[k], bool)]
    struct = [k for k in nonnum if fa[k] != fb[k]]
    num = [abs(float(fa[k]) - float(fb[k])) for k in set(fa) & set(fb) if k not in nonnum and fa[k] is not None and fb[k] is not None]
    return {"structure_identical": not keys_differ and not struct, "structure_differences": (keys_differ + struct)[:20],
            "max_abs_numeric_diff": max(num, default=0.0), "n_numeric": len(num)}


def materially_identical(a: Plan, b: Plan, tol: float = 1e-6) -> bool:
    c = compare(a, b)
    return c["structure_identical"] and c["max_abs_numeric_diff"] <= tol


if __name__ == "__main__":
    pa, pb = (Plan.model_validate_json(open(p).read()) for p in sys.argv[1:3])
    print(json.dumps(compare(pa, pb), indent=1))

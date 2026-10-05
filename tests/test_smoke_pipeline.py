"""End-to-end smoke test of the production execution path, independent of the (unshipped) sample
data: synthetic Stray export -> evaluator --execute -> roomscan.lidar.cli in a subprocess ->
plan.json -> scored against the exact synthetic geometry."""
import json

import yaml

from roomscan import synthetic
from roomscan.bench.evaluate import run


def test_lidar_capture_to_scored_plan(tmp_path):
    cap = synthetic.write(tmp_path / "syn", fps=10)
    site = tmp_path / "site"
    site.mkdir()
    T = synthetic.TRUTH
    rooms = [{"id": rid, "walls": [{"id": f"W{k + 1}", "length_m": L} for k, L in enumerate(T[rid]["walls"])],
              "ceiling_height_m": [T[rid]["ceiling"]], "floor_area_m2": T[rid]["area"]} for rid in ("A", "B")]
    rooms[0]["openings"] = [{"id": "d1", "kind": "doorway", "wall": "W2", "width_m": T["doorway_width"], "connects": "B"}]
    (site / "site.yaml").write_text(yaml.safe_dump({
        "site_id": "synthetic_two_rooms", "gt_kind": "synthetic", "purpose": "synthetic", "rooms": rooms,
        "captures": [{"id": "lidar", "tier": "lidar", "path": str(cap)}]}))
    rep = run(str(site), str(tmp_path / "run"), do_execute=True)
    assert rep["execution"]["lidar"]["status"] == "ok", rep["execution"]
    res = rep["captures"][0]
    assert res["rooms_matched"] == 2
    walls = [w for w in res["walls"] if w.get("pred") is not None]
    assert len(walls) == 8 and max(abs(w["abs_err"]) for w in walls) < 0.01
    assert max(abs(c["abs_err"]) for c in res["ceilings"]) < 0.01
    assert all(c["covered"] for c in res["ceilings"])
    assert res["adjacency"]["correct"]
    plan = json.loads((tmp_path / "run" / "lidar" / "plan.json").read_text())
    assert plan["quality_flags"] == []
    import jsonschema
    from pathlib import Path
    schema = json.loads((Path(__file__).resolve().parents[1] / "schema/plan.schema.json").read_text())
    jsonschema.validate(plan, schema)                    # the published schema file, not only the pydantic model

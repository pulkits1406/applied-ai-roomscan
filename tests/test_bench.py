"""Evaluator invariants on a synthetic site with known answers."""
import json

import pytest
import yaml
from pydantic import ValidationError

from roomscan.bench.evaluate import run
from roomscan.model import Measurement


def M(v, half, unit="m", status="measured"):
    return {"value": v, "unit": unit, "interval": {"lo": v - half, "hi": v + half, "level": 0.9}, "status": status, "method": "synthetic"}


def rect_plan(room_id, label, w, d, h, x0=0.0, openings=(), wall_shift=0):
    poly = [(x0, 0), (x0 + w, 0), (x0 + w, d), (x0, d)]
    lens = [w, d, w, d]
    surfaces, wall_ids = [], []
    for k in range(4):
        k2 = (k + wall_shift) % 4
        sid = f"{room_id}_w{k2}"
        wall_ids.append(sid)
        a, b = poly[k2], poly[(k2 + 1) % 4]
        surfaces.append({"id": sid, "kind": "wall", "room_id": room_id, "plane": {"normal": (0, 0, 1), "offset": 0},
                         "polygon": [(a[0], a[1], 0), (b[0], b[1], 0), (b[0], b[1], h), (a[0], a[1], h)],
                         "area": M(lens[k2] * h, 0.1, "m2"), "length": M(lens[k2], 0.02)})
    ops = [{"id": f"{room_id}_o{i}", "kind": "door", "wall_id": wall_ids[0], "room_ids": [room_id], "width": M(wd, 0.02), "detection_confidence": 0.9}
           for i, wd in enumerate(openings)]
    room = {"id": room_id, "label": label, "polygon": poly, "floor_area": M(w * d, 0.2, "m2"), "perimeter": M(2 * (w + d), 0.05),
            "ceiling_height": M(h, 0.01), "wall_ids": wall_ids, "opening_ids": [o["id"] for o in ops]}
    return room, surfaces, ops


def write_site(tmp_path, captures):
    site = {"site_id": "synthetic", "gt_kind": "laser", "purpose": "synthetic", "rooms": [
        {"id": "kitchen", "walls": [{"id": "W1", "length_m": [3.0, 3.001, 2.999]}, {"id": "W2", "length_m": 4.0}, {"id": "W3", "length_m": 3.0}, {"id": "W4", "length_m": 4.0}],
         "ceiling_height_m": [2.5, 2.502, 2.498], "openings": [{"id": "D1", "kind": "door", "wall": "W1", "width_m": 0.8, "connects": "hall"}]},
        {"id": "hall", "walls": [{"id": "W1", "length_m": 1.0}, {"id": "W2", "length_m": 4.0}, {"id": "W3", "length_m": 1.0}, {"id": "W4", "length_m": 4.0}],
         "ceiling_height_m": [2.5], "openings": [{"id": "D1", "kind": "door", "wall": "W1", "width_m": 0.8, "connects": "kitchen"}]}],
        "captures": captures}
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "site.yaml").write_text(yaml.safe_dump(site))
    return tmp_path


def write_plan(run_dir, cap_id, rooms, tier="lidar", adjacency=()):
    rs, ss, os_ = [], [], []
    for r, s, o in rooms:
        rs.append(r); ss += s; os_ += o
    plan = {"capture": {"tier": tier}, "rooms": rs, "surfaces": ss, "openings": os_, "adjacency": list(adjacency),
            "footprint_area": M(sum(r["floor_area"]["value"] for r in rs), 0.5, "m2")}
    (run_dir / cap_id).mkdir(parents=True, exist_ok=True)
    (run_dir / cap_id / "plan.json").write_text(json.dumps(plan))


def test_measurement_requires_interval():
    with pytest.raises(ValidationError):
        Measurement(value=1.0, unit="m", interval=None, method="x")
    Measurement(value=None, unit="m", interval=None, status="not_observed", method="x")


def test_perfect_plan_passes_and_wall_order_is_recovered(tmp_path):
    site = write_site(tmp_path / "site", [{"id": "a", "tier": "lidar", "path": "x", "rooms": ["kitchen", "hall"], "repeat_group": "g"},
                                         {"id": "b", "tier": "lidar", "path": "y", "rooms": ["kitchen", "hall"], "repeat_group": "g"}])
    runs = tmp_path / "run"
    adj = [{"room_a": "r1", "room_b": "r2"}]
    write_plan(runs, "a", [rect_plan("r1", None, 3.0, 4.0, 2.5, openings=[0.8], wall_shift=2), rect_plan("r2", None, 1.0, 4.0, 2.5, x0=3.2, openings=[0.81])], adjacency=adj)
    write_plan(runs, "b", [rect_plan("r1", None, 3.004, 4.0, 2.505, openings=[0.8]), rect_plan("r2", None, 1.0, 4.0, 2.5, x0=3.2, openings=[0.8])], adjacency=adj)
    rep = run(str(site), str(runs))
    a = next(c for c in rep["captures"] if c["capture"] == "a")
    assert all(abs(w["abs_err"]) < 1e-9 for w in a["walls"])  # cyclic shift undone
    assert a["opening_counts"] == {"gt": 2, "hits": 2, "missed": 0, "phantom": 0}
    assert a["adjacency"]["correct"]
    assert rep["gates"]["a"]["ceiling_pass"] and rep["gates"]["a"]["opening_pass"]
    rp = rep["gates"]["repeatability"]["g"]
    assert rp["pass_frac"] == 1.0


def test_phantom_and_missed_openings_count_against_gate(tmp_path):
    site = write_site(tmp_path / "site", [{"id": "a", "tier": "lidar", "path": "x", "rooms": ["kitchen", "hall"]}])
    runs = tmp_path / "run"
    write_plan(runs, "a", [rect_plan("r1", None, 3.0, 4.0, 2.5, openings=[0.8, 1.5]), rect_plan("r2", None, 1.0, 4.0, 2.53, x0=3.2)])
    rep = run(str(site), str(runs))
    a = rep["captures"][0]
    assert a["opening_counts"] == {"gt": 2, "hits": 1, "missed": 1, "phantom": 1}
    assert a["opening_hit_rate"] == pytest.approx(1 / 3)
    assert rep["gates"]["a"]["ceiling_pass"] is False  # 3 cm ceiling error in hall


def test_pseudo_gt_is_labelled(tmp_path):
    site = write_site(tmp_path / "site", [{"id": "a", "tier": "photo", "path": "x", "rooms": ["kitchen"]}])
    y = yaml.safe_load((site / "site.yaml").read_text()); y["gt_kind"] = "pseudo_lidar"; (site / "site.yaml").write_text(yaml.safe_dump(y))
    runs = tmp_path / "run"
    write_plan(runs, "a", [rect_plan("r1", "kitchen", 3.2, 4.2, 2.5)], tier="photo")
    rep = run(str(site), str(runs))
    assert rep["pseudo_gt"] is True
    assert "PSEUDO-GT" in (runs / "eval" / "synthetic.md").read_text()
    assert 0 < rep["gates"]["a"]["walls_within_gate_frac"] <= 1


def test_topology_mismatch_is_a_failure_not_a_fake_error(tmp_path):
    site = write_site(tmp_path / "site", [{"id": "a", "tier": "video", "path": "x", "rooms": ["kitchen"], "room_map": {"r1": "kitchen"}}])
    runs = tmp_path / "run"
    room, surfaces, ops = rect_plan("r1", None, 3.0, 4.0, 2.5)
    extra = dict(surfaces[0], id="r1_w9")
    room["wall_ids"].append("r1_w9")  # 5 predicted walls vs 4 GT walls
    write_plan(runs, "a", [(room, surfaces + [extra], ops)], tier="video")
    rep = run(str(site), str(runs))
    a = rep["captures"][0]
    assert a["topology_mismatch"] == [{"room": "kitchen", "gt_walls": 4, "pred_walls": 5}]
    assert all(w.get("pred") is None for w in a["walls"])
    assert rep["gates"]["a"]["walls_within_gate_frac"] == 0.0


def test_calibrated_interval_requires_reference():
    from roomscan.model import Interval
    with pytest.raises(ValidationError):
        Interval(lo=0, hi=1, calibrated=True)
    assert Interval(lo=0, hi=1).calibrated is False


def test_residual_rows_are_tagged_with_gt_kind(tmp_path):
    site = write_site(tmp_path / "site", [{"id": "a", "tier": "lidar", "path": "x", "rooms": ["kitchen"], "room_map": {"r1": "kitchen"}}])
    runs = tmp_path / "run"
    write_plan(runs, "a", [rect_plan("r1", None, 3.01, 4.0, 2.5)])
    rep = run(str(site), str(runs))
    rows = rep["calibration_residuals"]
    assert rows and all(r["gt_kind"] == "laser" for r in rows)
    assert {r["quantity"] for r in rows} >= {"wall_length", "ceiling_height", "floor_area"}

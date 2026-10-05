"""Damage regions, concealed-damage rules and scope items on a measured plan (mechanism only:
no detector, no damage benchmark)."""
import numpy as np
import pytest

from roomscan.assemble import OpeningInput, RoomInput, assemble
from roomscan.damage import attach
from roomscan.model import CaptureInfo, Plan

RECT = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], float)


def _plan():
    rooms = [RoomInput("r1", RECT, [0.007] * 4, ["measured"] * 4, 0.004, 2.5, 0.007, "t", label="bedroom"),
             RoomInput("r2", RECT + [4.1, 0], [0.007] * 4, ["measured"] * 4, 0.004, 2.5, 0.007, "t", label="bathroom")]
    ops = [OpeningInput("o0", "doorway", ["r1", "r2"], 1, 0.8, 0.01, "t", 0.8)]
    return assemble("lidar", rooms, ops, CaptureInfo(tier="lidar"), "t", [])


def test_rules_fire_with_ids_and_scope_is_keyed_to_surfaces():
    plan = _plan()
    obs = {"source": "manual", "observations": [
        {"surface_id": "r1_ceiling", "class": "water_stain", "region_uv": [[1, 1], [1.5, 1], [1.5, 1.4], [1, 1.4]]},
        {"surface_id": "r1_w1", "class": "mould", "region_uv": [[0.5, 0.0], [1.0, 0.0], [1.0, 0.2], [0.5, 0.2]]},
        {"surface_id": "r1_w0", "class": "crack", "region_uv": [[0.2, 1.0], [1.5, 1.6], [1.5, 1.62]]}]}
    out = attach(plan, obs)
    Plan.model_validate(out.model_dump())                                  # schema-valid, references resolve
    assert [d.damage_class for d in out.damage] == ["water_stain", "mould", "crack"]
    assert abs(out.damage[0].area.value - 0.2) < 1e-9 and out.damage[0].area.interval.lo < 0.2 < out.damage[0].area.interval.hi
    rules = {f.rule_id for f in out.concealed_damage_flags}
    assert {"CD1", "CD2", "CD3", "CD4"} <= rules                           # wall r1_w1 holds the doorway to the bathroom
    cd3 = [f for f in out.concealed_damage_flags if f.rule_id == "CD3"]
    assert [f.surface_ids for f in cd3] == [["r1_w1"]]                     # not the other walls of r1
    sids = {s.id for s in out.surfaces}
    assert out.scope and all(s.surface_id in sids for s in out.scope)
    assert any(s.quantity.unit == "m" for s in out.scope)                  # crack length


def test_unknown_surface_or_class_is_rejected():
    with pytest.raises(ValueError):
        attach(_plan(), {"observations": [{"surface_id": "nope", "class": "mould", "region_uv": [[0, 0], [1, 0], [1, 1]]}]})
    with pytest.raises(ValueError):
        attach(_plan(), {"observations": [{"surface_id": "r1_w0", "class": "rust", "region_uv": [[0, 0], [1, 0], [1, 1]]}]})

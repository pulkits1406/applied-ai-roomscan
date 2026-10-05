"""Development-site evaluation: partial tape GT, question grouping, scale/shape split, purpose
rules and the tuning-leakage guard."""
import pytest
import yaml

from roomscan import synthetic
from roomscan.bench import leakage
from roomscan.bench.evaluate import run
from roomscan.bench.gt import load_site


def _site(tmp_path, purpose, cap_id, cap_path):
    d = tmp_path / f"site_{purpose}"
    d.mkdir()
    rooms = [{"id": "A", "walls": [{"id": "W1", "length_m": 3.0}, {"id": "W2", "length_m": [4.0, 4.001]}, {"id": "W3"}, {"id": "W4"}],
              "ceiling_height_m": [2.5], "openings": [{"id": "D1", "kind": "doorway", "wall": "W2", "width_m": 0.8, "connects": "B"}]},
             {"id": "B", "walls": [{"id": f"W{k}"} for k in range(1, 5)]}]
    (d / "site.yaml").write_text(yaml.safe_dump({"site_id": f"{purpose}_x", "gt_kind": "tape", "purpose": purpose, "rooms": rooms,
                                                 "captures": [{"id": cap_id, "tier": "lidar", "path": str(cap_path), "question": "A",
                                                               "variant": "protocol"}]}))
    return d


def test_dev_site_end_to_end_with_partial_tape(tmp_path):
    cap = synthetic.write(tmp_path / "syn", fps=10)
    rep = run(str(_site(tmp_path, "dev", "dev_A_lidar", cap)), str(tmp_path / "run"), do_execute=True)
    res = rep["captures"][0]
    walls = [w for w in res["walls"] if w.get("pred") is not None]
    assert len(walls) == 2 and max(abs(w["abs_err"]) for w in walls) < 0.01        # only the two taped walls are scored
    a = next(d for d in res["rooms_detail"] if d["room"] == "A")
    assert abs(a["scale_ratio"] - 1) < 0.005 and a["shape_err_median"] < 0.005
    md = (tmp_path / "run" / "eval" / "dev_x.md").read_text()
    assert "DEVELOPMENT DATA" in md and "## Question A" in md
    assert (tmp_path / "run" / "eval" / "dev_x_walls.png").exists()
    leakage.check([rep], "tune")
    with pytest.raises(leakage.LeakageError):
        leakage.check([rep], "benchmark_report")


def test_purpose_rules(tmp_path):
    with pytest.raises(ValueError, match="prefixed"):
        load_site(_site(tmp_path, "dev", "lidar_1", tmp_path))                      # dev site, unprefixed capture
    with pytest.raises(ValueError):
        load_site(_site(tmp_path, "bm", "dev_A_lidar", tmp_path))                  # dev capture in a benchmark site
    rep = {"site": "bm_x", "purpose": "bm", "gt_kind": "laser"}
    with pytest.raises(leakage.LeakageError):
        leakage.check([rep], "tune")
    with pytest.raises(leakage.LeakageError):
        leakage.check([{"site": "p", "purpose": "pseudo", "gt_kind": "pseudo_lidar"}], "calibrate")


def test_template_dev_site_is_valid():
    s = load_site("benchmark/_template_dev")
    assert s.purpose == "dev" and all(c.id.startswith("dev_") for c in s.captures)
    assert {c.question for c in s.captures} == {"A", "B", "C"}

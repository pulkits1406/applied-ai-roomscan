"""Benchmark report from machine-generated results only.

    uv run python reports/build_reports.py --regenerate   # recompute every result from raw inputs (~25 min, GPU for photo/video)
    uv run python reports/build_reports.py                # re-render reports/benchmark_report.md from reports/data/*.json

--regenerate needs the supplied sample captures unzipped at the repo root and the frame cache
(scripts/prepare_intermediates.sh); it writes runs/report/ (ignored) and copies the JSON results
into reports/data/ (tracked). Rendering reads reports/data/ only, so every number in the report
comes from a file a script produced.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/data"
RUN = ROOT / "runs/report"
CAP = "single_scan_with_ceiling"


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)


def regenerate():
    DATA.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    uv = ["uv", "run"]
    sh(*uv, "roomscan-lidar", CAP, RUN / "lidar_with_ceiling")
    sh(*uv, "roomscan-lidar", CAP, RUN / "lidar_with_ceiling_arkit", "--placement", "arkit")
    sh(*uv, "python", "experiments/e24_three_tier_eval/make_site.py", RUN / "lidar_with_ceiling/plan.json")
    sh(*uv, "python", "experiments/e23_photo_frontend/make_photo_folders.py", CAP, "6")
    sh(*uv, "python", "experiments/e23_photo_frontend/make_photo_folders.py", CAP, "8", "", "sweep")
    sh(*uv, "roomscan-bench", "benchmark/pseudo_gt_lidar_v11", "--run", RUN / "three_tier", "--execute", "--force")
    sh(*uv, "python", "experiments/e24_three_tier_eval/video_identity.py", RUN / "three_tier", CAP, RUN / "lidar_with_ceiling/plan.json")
    sh(*uv, "python", "experiments/e22_lidar_small_rooms/repeat.py", CAP, "v1_0,ladder")
    sh("reports/fix_loop/run.sh")
    shutil.copy(RUN / "three_tier/eval/pseudo_gt_lidar_v11.json", DATA / "pseudo_gt_eval.json")
    shutil.copy(RUN / "three_tier/eval/pseudo_gt_lidar_v11_walls.png", DATA / "pseudo_gt_walls.png")
    shutil.copy(ROOT / "experiments/e24_three_tier_eval/out/video_identity.json", DATA / "video_identity.json")
    shutil.copy(ROOT / f"experiments/e22_lidar_small_rooms/out/repeat_{CAP}_v1_0_ladder.json", DATA / "repeatability.json")
    shutil.copy(ROOT / f"experiments/e19_drift_stitch/out/ablation_v2_{CAP}_V2.json", DATA / "drift_e19.json")
    for c in ("lidar_floor_only", "lidar_single_room", "photo_sweep_k8", "photo_diverse_k6", "video_with_ceiling"):
        shutil.copy(RUN / "three_tier" / c / "plan.png", DATA / f"plan_{c}.png")
    shutil.copy(RUN / "lidar_with_ceiling/plan.png", DATA / "plan_lidar_with_ceiling.png")
    placement(RUN / "lidar_with_ceiling/plan.json", RUN / "lidar_with_ceiling_arkit/plan.json")
    timing()


def placement(on_path, off_path):
    """Drift ON/OFF on the production pipeline: arkit_yaw (common Manhattan yaw) vs raw ARKit poses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
    out, fig = {}, plt.figure(figsize=(7, 7))
    ax = fig.gca()
    for key, p, col in (("on_arkit_yaw", on_path, "tab:blue"), ("off_raw_arkit", off_path, "tab:red")):
        plan = json.loads(Path(p).read_text())
        polys = {r["id"]: Polygon(r["polygon"]).buffer(0) for r in plan["rooms"]}
        ids = sorted(polys)
        ov = [polys[a].intersection(polys[b]).area for i, a in enumerate(ids) for b in ids[i + 1:]]
        out[key] = {"rooms": len(ids), "sum_room_area_m2": round(sum(x.area for x in polys.values()), 3),
                    "union_footprint_m2": round(unary_union(list(polys.values())).area, 3),
                    "total_overlap_m2": round(float(sum(ov)), 3), "max_pair_overlap_m2": round(float(max(ov, default=0)), 3),
                    "centroids": {k: [round(v.centroid.x, 3), round(v.centroid.y, 3)] for k, v in polys.items()}}
        for k, poly in polys.items():
            x, y = poly.exterior.xy
            ax.plot(x, y, color=col, lw=1.2, ls="-" if key.startswith("on") else "--")
    out["centroid_shift_m"] = {k: round(float(np.linalg.norm(np.subtract(out["on_arkit_yaw"]["centroids"][k], out["off_raw_arkit"]["centroids"][k]))), 3)
                               for k in out["on_arkit_yaw"]["centroids"] if k in out["off_raw_arkit"]["centroids"]}
    ax.set_aspect("equal"); ax.set_title("Stitched footprint: drift handling ON (arkit_yaw, solid) vs OFF (raw ARKit, dashed)", fontsize=9)
    fig.tight_layout(); fig.savefig(DATA / "drift_on_off.png", dpi=120); plt.close(fig)
    (DATA / "drift_on_off.json").write_text(json.dumps(out, indent=1))


def timing():
    ev = json.loads((DATA / "pseudo_gt_eval.json").read_text())
    t = {k: v.get("seconds") for k, v in (ev.get("execution") or {}).items()}
    for name in ("lidar_with_ceiling", "lidar_with_ceiling_arkit"):
        d = RUN / name / "debug.json"
        if d.exists():
            t[name] = json.loads(d.read_text()).get("seconds")
    (DATA / "timing.json").write_text(json.dumps(t, indent=1))


# ------------------------------------------------------------------ rendering
def _f(x, nd=3):
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render() -> str:
    ev = json.loads((DATA / "pseudo_gt_eval.json").read_text())
    vid = json.loads((DATA / "video_identity.json").read_text())
    rep = json.loads((DATA / "repeatability.json").read_text())
    e19 = json.loads((DATA / "drift_e19.json").read_text())
    dr = json.loads((DATA / "drift_on_off.json").read_text())
    tm = json.loads((DATA / "timing.json").read_text())
    fl = json.loads((ROOT / "reports/fix_loop/results/fix_loop_synthetic.json").read_text())
    L = ["# Benchmark report", "",
         "Generated by `reports/build_reports.py` from `reports/data/*.json` (regenerate with `--regenerate`). "
         "No number below was typed by hand.", "",
         "> **There is no laser/tape benchmark yet** (no iPhone available): no result here is benchmark accuracy. "
         "Sections are labelled PSEUDO-GT (scored against our own LiDAR reconstruction of the supplied capture; LiDAR and the "
         "ARKit trajectory disagree by ~3 %, e08), GT-FREE (internal consistency), or SYNTHETIC (rendered exact geometry: tests code, not sensors).", ""]
    # ---- pseudo-GT per tier
    L += ["## 1. Per-tier results on the supplied data [PSEUDO-GT]", "",
          f"Site `{ev['site']}` (purpose `{ev['purpose']}`, gt_kind `{ev['gt_kind']}`): reference = LiDAR v1.1 plan of `{CAP}`; "
          f"code {', '.join(c[:7] for c in ev.get('code_versions') or [])}{' (dirty tree)' if ev.get('dirty_code') else ''}. "
          "The reference capture is never scored against itself. Photo folders are simulated from walkthrough frames (e23).", "",
          "| capture | tier | rooms matched | match | walls scored | walls within tier gate | median \\|wall err\\| | median \\|area err\\| | interval coverage | flags |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    gate_txt = {"photo": "±8 %", "video": "±3 %", "lidar": "no relative gate"}
    for c in ev["captures"]:
        g = ev["gates"].get(c["capture"], {})
        w = [x for x in c["walls"] if x.get("pred") is not None]
        meth = sorted({m["method"] for m in c.get("room_match", [])}) or ["—"]
        cov = [x["covered"] for k in ("walls", "ceilings", "areas", "openings") for x in c[k] if x.get("covered") is not None]
        within = g.get("walls_within_gate_frac")
        L.append(f"| {c['capture']} | {c['tier']} | {c.get('rooms_matched')}/{c.get('rooms_expected')} | {'/'.join(meth)} | {len(w)} | "
                 f"{'—' if within is None else f'{100 * within:.0f} % ({gate_txt[c["tier"]]})'} | "
                 f"{_f(g.get('wall_rel_err_median') and 100 * g['wall_rel_err_median'], 1)} % | {_f(g.get('area_rel_err_median') and 100 * g['area_rel_err_median'], 1)} % | "
                 f"{sum(cov)}/{len(cov)} | {len(c.get('quality_flags', []))} |")
    L += ["", "Size-matched rooms (`perimeter`) have unverified identity. Video rooms by true identity (`video_identity.py`, ARKit room labels "
          "used for evaluation only):", "", "| video room | time span (s) | area (m²) | true room | true area (m²) | area error |", "|---|---|---|---|---|---|"]
    for v in vid:
        L.append(f"| {v['room']} | {v['span_s'][0]:.0f}–{v['span_s'][1]:.0f} | {v['area']} | {v['truth_room']} | {v['truth_area']} | "
                 f"{_f(v['area_rel_err_by_identity'] and 100 * v['area_rel_err_by_identity'], 1)} % |")
    L += ["", "![walls](data/pseudo_gt_walls.png)", ""]
    # ---- repeatability
    L += ["## 2. LiDAR repeatability [GT-FREE]", "",
          "Production code, e18 protocol: topology from all data, wall lengths re-measured from two disjoint 2 s-chunk halves of the same "
          "capture (each ~50 % of the data). **Not the repeatability gate**, which needs two separate captures (blocked).", "",
          "| config | rooms with outline | walls | wall Δ median (mm) | wall Δ p90 (mm) | walls within max(1 cm, 0.5 %) | ceilings Δ ≤ 1 cm |",
          "|---|---|---|---|---|---|---|"]
    for k, s in rep["summary"].items():
        L.append(f"| {k} | {s['rooms_with_polygon']} | {s['n_walls']} | {s['wall_diff_mm_median']} | {s['wall_diff_mm_p90']} | "
                 f"{_f(s['wall_pass_frac'] and 100 * s['wall_pass_frac'], 0)} % | {_f(s['ceiling_pass_frac_1cm'] and 100 * s['ceiling_pass_frac_1cm'], 0)} % |")
    # ---- drift
    L += ["", "## 3. Drift ON/OFF [GT-FREE]", "",
          "Drift handling = room-local measurement (each room from its own registered visits) + common-Manhattan-yaw placement "
          "(`arkit_yaw`); room translations are as ARKit placed them. Production pipeline, ON vs OFF (raw ARKit placement):", "",
          "| placement | rooms | sum of room areas (m²) | union footprint (m²) | total room overlap (m²) | max pair overlap (m²) |", "|---|---|---|---|---|---|"]
    for k in ("on_arkit_yaw", "off_raw_arkit"):
        d = dr[k]
        L.append(f"| {k} | {d['rooms']} | {d['sum_room_area_m2']} | {d['union_footprint_m2']} | {d['total_overlap_m2']} | {d['max_pair_overlap_m2']} |")
    L += ["", f"Room centroid shift ON vs OFF (m): {dr['centroid_shift_m']}", "", "![drift](data/drift_on_off.png)", "",
          "Placement disagreement between two disjoint halves of the capture after a rigid fit (e19, GT-free; corners, mm):", "",
          "| stitch variant | corner median | corner p90 | corner max |", "|---|---|---|---|"]
    names = {"A": "raw ARKit", "B1": "+ common yaw (production)", "B2": "+ doorway constraints (none measurable)",
             "B3": "+ shared walls, one thickness", "B4": "+ shared walls, per-wall thickness"}
    for k in ("A", "B1", "B2", "B3", "B4"):
        c = e19[k]["corners"]
        L.append(f"| {k} {names[k]} | {c['median_mm']} | {c['p90_mm']} | {c['max_mm']} |")
    # ---- synthetic
    L += ["", "## 4. Synthetic validation [SYNTHETIC — not benchmark evidence]", "",
          "Exact rendered two-room capture (`src/roomscan/synthetic.py`), LiDAR tier, scored by the same evaluator (fix-loop site):", "",
          "| run | rooms | wall errors (mm) | ceiling errors (mm) | doorway error (mm) |", "|---|---|---|---|---|"]
    for c in fl["captures"]:
        w = [x for x in c["walls"] if x.get("pred") is not None]
        we = ", ".join(f"{1000 * x['abs_err']:+.1f}" for x in w) or "none matched"
        ce = ", ".join(f"{1000 * x['abs_err']:+.1f}" for x in c["ceilings"] if x.get("abs_err") is not None) or "—"
        de = ", ".join(f"{1000 * x['abs_err']:+.0f}" for x in c["openings"] if "abs_err" in x) or "—"
        L.append(f"| {c['capture']} | {c.get('rooms_matched')}/{c.get('rooms_expected')} | {we} | {ce} | {de} |")
    # ---- timing
    L += ["", "## 5. Timing (Apple M4, 16 GB, MPS) [MEASURED]", "", "| run | seconds |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in tm.items()]
    # ---- coverage & missing
    L += ["", "## 6. Interval coverage", "",
          "Coverage = fraction of reference values inside the stated 90 % intervals (uncalibrated) in section 1. Intervals widen with "
          "measured geometric instability (`src/roomscan/quality.py`), but rooms that are consistently wrong are reported by flags, not "
          "covered. Calibration requires laser GT (blocked).", "",
          "## 7. Missing real benchmark evidence (BLOCKED: no iPhone)", "",
          "- Laser/tape GT, the multi-room benchmark at all three tiers, the damage room, repeat captures: none exist.",
          "- Opening-width gate, ceiling gate, repeatability gate, photo stitch gate, photo/video wall gates on real captures: not evaluated.",
          "- Head-to-head vs a consumer app (magicplan/Polycam): no export exists.",
          "- Real-device file formats (Stray current version, HEIC, MOV) are tested only with synthetic files.",
          "- Walk-in readiness on an unseen space: untested.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regenerate", action="store_true")
    a = ap.parse_args()
    if a.regenerate:
        regenerate()
    (ROOT / "reports/benchmark_report.md").write_text(render())
    print("wrote reports/benchmark_report.md")


if __name__ == "__main__":
    sys.exit(main())

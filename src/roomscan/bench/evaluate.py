"""Score predicted plans against a benchmark site; one command for the whole site.

    uv run roomscan-bench benchmark/<site> --run runs/<run_id> [--execute [--force]]

With --execute, every capture in site.yaml is first run through its tier's production pipeline
(roomscan.<tier>.cli, one subprocess per capture: raw input -> runs/<run_id>/<capture_id>/plan.json);
existing plans are reused unless --force. Without it, the plans must already exist. Writes
runs/<run_id>/eval/<site_id>.json and .md. Gate thresholds are those of the assessment.
Results from a site with gt_kind == "pseudo_lidar" are labelled PSEUDO-GT everywhere.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

from roomscan.bench import diagnostics
from roomscan.bench.gt import Site, load_site
from roomscan.bench.match import match_openings, match_rooms, match_walls
from roomscan.model import Measurement, Plan

GATES = {
    "opening_abs_m": 0.02, "opening_hit_rate": 0.85,
    "ceiling_abs_m": 0.015, "ceiling_repeat_spread_m": 0.01,
    "repeat_abs_m": 0.01, "repeat_rel": 0.005,
    "wall_rel": {"photo": 0.08, "video": 0.03, "lidar": None},
    "footprint_rel_photo": 0.08,
}


def _cover(m: Measurement | None, truth: float) -> bool | None:
    if m is None or m.interval is None:
        return None
    return bool(m.interval.lo <= truth <= m.interval.hi)


def evaluate_capture(site: Site, cap, plan: Plan) -> dict:
    rmap = match_rooms(plan, [site.room(r) for r in (cap.rooms or [g.id for g in site.rooms])], cap.room_map)
    labels = {r.id: r.label for r in plan.rooms}
    # identity is known from an explicit map or a matching label; perimeter matching only pairs similar sizes
    match_method = {pid: ("map" if pid in cap.room_map else "label" if labels.get(pid) == gid else "perimeter") for pid, gid in rmap.items()}
    surf = {s.id: s for s in plan.surfaces}
    walls, ceilings, opening_rows, areas, topology_mismatch, rooms_detail = [], [], [], [], [], []
    hits = misses = phantoms = 0
    for pr in plan.rooms:
        if pr.id not in rmap:
            continue
        g = site.room(rmap[pr.id])
        wdet = {}
        wm = match_walls(plan, pr, g, wdet)
        if len(pr.wall_ids) != len(g.walls):
            topology_mismatch.append({"room": g.id, "gt_walls": len(g.walls), "pred_walls": len(pr.wall_ids)})
        ratios = []
        for gw in g.walls:
            if gw.length is None:
                continue                      # not measured (partial tape GT)
            if gw.id not in wm:
                walls.append({"room": g.id, "wall": gw.id, "gt": gw.length, "pred": None, "unmatched": True})
                continue
            m = surf[wm[gw.id]].length
            walls.append({"room": g.id, "wall": gw.id, "gt": gw.length, "pred": m.value, "abs_err": m.value - gw.length,
                          "rel_err": (m.value - gw.length) / gw.length, "covered": _cover(m, gw.length), "status": m.status,
                          "half_width": None if m.interval is None else m.interval.hi - m.value})
            ratios.append(m.value / gw.length)
        det = {"room": g.id, "pred": pr.id, "match": match_method[pr.id], "pred_walls": len(pr.wall_ids), "gt_walls": len(g.walls),
               "wall_match": wdet, "placed": pr.placed}
        if len(ratios) >= 2:
            sc = float(np.median(ratios))
            # scale vs shape: a common factor (scale) and what remains after removing it (shape)
            det.update({"scale_ratio": sc, "shape_err_median": float(np.median(np.abs(np.array(ratios) / sc - 1)))})
        rooms_detail.append(det)
        if g.ceiling_height is not None:
            m = pr.ceiling_height
            ceilings.append({"room": g.id, "gt": g.ceiling_height, "pred": m.value, "status": m.status,
                             "abs_err": None if m.value is None else m.value - g.ceiling_height, "covered": _cover(m, g.ceiling_height)})
        if g.area is not None:
            areas.append({"room": g.id, "gt": g.area, "pred": pr.floor_area.value, "rel_err": pr.floor_area.value / g.area - 1,
                          "covered": _cover(pr.floor_area, g.area)})
        pairs, mg, mp = match_openings(plan, pr, g, wm)
        for go, po in pairs:
            err = po.width.value - go.width
            hit = abs(err) <= GATES["opening_abs_m"]
            hits += hit
            opening_rows.append({"room": g.id, "opening": go.id, "gt": go.width, "pred": po.width.value, "abs_err": err, "hit": hit,
                                 "covered": _cover(po.width, go.width)})
        misses += len(mg); phantoms += len(mp)
        opening_rows += [{"room": g.id, "opening": x, "missed": True} for x in mg]
        opening_rows += [{"room": g.id, "pred_opening": x, "phantom": True} for x in mp]
    gt_open = sum(len(site.room(rmap[p.id]).openings) for p in plan.rooms if p.id in rmap)
    # stitch: adjacency + overlaps
    inv = rmap
    pred_adj = {frozenset((inv.get(a.room_a), inv.get(a.room_b))) for a in plan.adjacency}
    gt_adj = {frozenset((g.id, o.connects)) for g in site.rooms for o in g.openings if o.connects in {r.id for r in site.rooms}}
    covered_rooms = set(rmap.values())
    gt_adj = {e for e in gt_adj if e <= covered_rooms}
    placed = [r for r in plan.rooms if r.placed]
    polys = [Polygon(r.polygon) for r in placed]
    overlap = max([a.intersection(b).area for a, b in combinations(polys, 2)], default=0.0)
    out = {"capture": cap.id, "tier": cap.tier, "rooms_matched": len(rmap), "rooms_expected": len(cap.rooms or site.rooms),
           "room_match": [{"pred": p, "gt": g, "method": match_method[p]} for p, g in rmap.items()],
           "rooms_detail": rooms_detail, "phantom_rooms": [r.id for r in plan.rooms if r.id not in rmap],
           "walls": walls, "ceilings": ceilings, "areas": areas, "openings": opening_rows,
           "opening_hit_rate": None if not any(r.openings for r in site.rooms) else (hits / max(gt_open + phantoms, 1) if (gt_open + phantoms) else None),
           "topology_mismatch": topology_mismatch,
           "opening_counts": {"gt": gt_open, "hits": hits, "missed": misses, "phantom": phantoms},
           "adjacency": {"correct": pred_adj == gt_adj, "missing": [sorted(e) for e in gt_adj - pred_adj], "extra": [sorted(map(str, e)) for e in pred_adj - gt_adj]},
           "max_room_overlap_m2": overlap, "rooms_unplaced": len(plan.rooms) - len(placed), "quality_flags": plan.quality_flags}
    if areas:
        gt_fp = sum(a["gt"] for a in areas); pr_fp = sum(a["pred"] for a in areas)
        out["footprint_rel_err"] = pr_fp / gt_fp - 1
    return out


def gates(site: Site, results: list[dict]) -> dict:
    g = {}
    for r in results:
        t = r["tier"]; key = r["capture"]
        werr = [abs(w["rel_err"]) for w in r["walls"] if w.get("pred") is not None]
        n_unmatched = sum(1 for w in r["walls"] if w.get("pred") is None)
        row = {"tier": t, "rooms_matched": f"{r.get('rooms_matched', 0)}/{r.get('rooms_expected', '?')}", "walls_matched": len(werr),
               "walls_unmatched": n_unmatched, "rooms_topology_mismatch": len(r.get("topology_mismatch", []))}
        n_perim = sum(1 for m in r.get("room_match", []) if m["method"] == "perimeter")
        if n_perim:
            row["rooms_matched_by_size_only"] = f"{n_perim} (identity unverified: errors may be optimistic)"
        if werr:
            row["wall_rel_err_median"] = float(np.median(werr))
        aerr = [abs(a["rel_err"]) for a in r.get("areas", []) if a.get("pred") is not None]
        if aerr:
            row["area_rel_err_median"] = float(np.median(aerr))
        cerr = [abs(c["abs_err"]) for c in r.get("ceilings", []) if c.get("abs_err") is not None]
        if cerr:
            row["ceiling_abs_err_median_m"] = float(np.median(cerr))
        if GATES["wall_rel"][t] is not None and (werr or n_unmatched):
            # unmatched GT walls count as failures
            row["walls_within_gate_frac"] = float(np.sum([e <= GATES["wall_rel"][t] for e in werr]) / (len(werr) + n_unmatched))
            row["walls_max_rel_err"] = float(max(werr)) if werr else None
        if t == "lidar":
            ce = [abs(c["abs_err"]) for c in r["ceilings"] if c["abs_err"] is not None]
            if ce: row["ceiling_pass"] = bool(max(ce) <= GATES["ceiling_abs_m"]); row["ceiling_max_abs_err_m"] = float(max(ce))
            row["opening_pass"] = "not_evaluated (no GT openings)" if r["opening_hit_rate"] is None else r["opening_hit_rate"] >= GATES["opening_hit_rate"]
        if t == "photo":
            row["stitch_pass"] = bool(r["adjacency"]["correct"] and r["max_room_overlap_m2"] < 0.05 and not r.get("rooms_unplaced")
                                      and abs(r.get("footprint_rel_err", 1)) <= GATES["footprint_rel_photo"])
            if r.get("rooms_unplaced"):
                row["stitch_fail_reason"] = f"{r['rooms_unplaced']} room(s) not placed"
        if r.get("missing_prediction"):
            row["missing_prediction"] = True
        cov = [x["covered"] for k in ("walls", "ceilings", "areas", "openings") for x in r[k] if x.get("covered") is not None]
        if cov: row["interval_coverage"] = float(np.mean(cov)); row["n_intervals"] = len(cov)
        g[key] = row
    # repeatability
    groups = {}
    for c in site.captures:
        if c.repeat_group: groups.setdefault(c.repeat_group, []).append(c.id)
    res = {r["capture"]: r for r in results}
    rep = {}
    for grp, ids in groups.items():
        ids = [i for i in ids if i in res]
        if len(ids) < 2: continue
        rows = []
        for a, b in combinations(ids, 2):
            wa = {(w["room"], w["wall"]): w["pred"] for w in res[a]["walls"] if w.get("pred") is not None}
            wb = {(w["room"], w["wall"]): w["pred"] for w in res[b]["walls"] if w.get("pred") is not None}
            for k in wa.keys() & wb.keys():
                d = abs(wa[k] - wb[k]); tol = max(GATES["repeat_abs_m"], GATES["repeat_rel"] * 0.5 * (wa[k] + wb[k]))
                rows.append({"pair": [a, b], "room": k[0], "wall": k[1], "diff_m": d, "tol_m": tol, "pass": d <= tol})
            ca = {c["room"]: c["pred"] for c in res[a]["ceilings"] if c["pred"] is not None}
            cb = {c["room"]: c["pred"] for c in res[b]["ceilings"] if c["pred"] is not None}
            for k in ca.keys() & cb.keys():
                rows.append({"pair": [a, b], "room": k, "ceiling_spread_m": abs(ca[k] - cb[k]), "pass": abs(ca[k] - cb[k]) <= GATES["ceiling_repeat_spread_m"]})
        rep[grp] = {"rows": rows, "pass_frac": float(np.mean([x["pass"] for x in rows])) if rows else None}
    g["repeatability"] = rep
    # head-to-head
    h2h = []
    for inc in site.incumbent:
        for rid, ir in inc.rooms.items():
            gt = site.room(rid)
            ours = {}
            for r in results:
                if r["tier"] != "lidar": continue
                for w in r["walls"]:
                    if w["room"] == rid and w.get("pred") is not None: ours[("wall", w["wall"])] = abs(w["abs_err"])
                for c in r["ceilings"]:
                    if c["room"] == rid and c["abs_err"] is not None: ours[("ceiling", rid)] = abs(c["abs_err"])
            for wid, v in ir.walls.items():
                gw = next(w for w in gt.walls if w.id == wid)
                if ("wall", wid) in ours: h2h.append({"app": inc.app, "room": rid, "dim": wid, "ours": ours[("wall", wid)], "theirs": abs(v - gw.length)})
            if ir.ceiling_height is not None and gt.ceiling_height is not None and ("ceiling", rid) in ours:
                h2h.append({"app": inc.app, "room": rid, "dim": "ceiling", "ours": ours[("ceiling", rid)], "theirs": abs(ir.ceiling_height - gt.ceiling_height)})
    if h2h:
        g["head_to_head"] = {"rows": h2h, "beat_or_tie_frac": float(np.mean([x["ours"] <= x["theirs"] + 1e-3 for x in h2h]))}
    return g


def calibration_residuals(site: Site, results: list[dict]) -> list[dict]:
    """Flat (tier, quantity, gt, pred, residual, covered) rows: the input an empirical interval
    calibration needs. Rows from pseudo-GT sites are tagged so they are never used to calibrate."""
    rows = []
    for r in results:
        for kind, key in (("wall_length", "walls"), ("ceiling_height", "ceilings"), ("floor_area", "areas"), ("opening_width", "openings")):
            for x in r.get(key, []):
                if x.get("pred") is None or x.get("gt") is None:
                    continue
                rows.append({"gt_kind": site.gt_kind, "purpose": site.purpose, "capture": r["capture"], "tier": r["tier"], "quantity": kind, "room": x.get("room"),
                             "gt": x["gt"], "pred": x["pred"], "residual": x["pred"] - x["gt"], "covered": x.get("covered")})
    return rows


BANNER = {
    "dev": "> **DEVELOPMENT DATA.** Answers protocol/model questions. Not benchmark evidence; not used for calibration unless named explicitly (bench/leakage.py).",
    "fail": "> **FAILURE-MODE DATA.** Deliberately poor captures: checks that failures are flagged, not accuracy.",
}


def _question_tables(results: list[dict]) -> list[str]:
    qs = sorted({r.get("question") for r in results if r.get("question")})
    lines = []
    for q in qs:
        lines += ["", f"## Question {q}", "",
                  "| capture | variant | rooms matched | phantom rooms | walls (measured, in interval) | median abs wall err | scale ratio | shape err | doorway err | diagnostics |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in [x for x in results if x.get("question") == q]:
            w = [x for x in r["walls"] if x.get("pred") is not None]
            err = f"{1000 * float(np.median([abs(x['abs_err']) for x in w])):.0f} mm" if w else "—"
            cov = f"{len(w)}, {sum(1 for x in w if x.get('covered'))}"
            rd = [d for d in r.get("rooms_detail", []) if "scale_ratio" in d]
            sc = ", ".join(f"{d['room']} {d['scale_ratio']:.3f}" for d in rd) or "—"
            sh = ", ".join(f"{d['room']} {100 * d['shape_err_median']:.1f} %" for d in rd) or "—"
            dw = [x for x in r["openings"] if "abs_err" in x]
            dws = ", ".join(f"{1000 * x['abs_err']:+.0f} mm" for x in dw) or ("missed" if any(x.get("missed") for x in r["openings"]) else "—")
            dg = r.get("diagnostics") or {}
            dgs = (f"windows {dg['windows_coherent']}/{dg['windows']}, longest run {dg['longest_coherent_run_windows']}" if "windows" in dg else
                   f"upper walls {dg.get('upper_walls_observed')}" if "upper_walls_observed" in dg else
                   "; ".join(f"{k}: h-spread {v.get('camera_height_spread_m')}" for k, v in dg.get("rooms", {}).items()) if dg else "—")
            lines.append(f"| {r['capture']} | {r.get('variant') or ''} | {r.get('rooms_matched')}/{r.get('rooms_expected')} | "
                         f"{len(r.get('phantom_rooms', []))} | {cov} | {err} | {sc} | {sh} | {dws} | {dgs} |")
    return lines


def render_md(site: Site, results: list[dict], gate: dict, report: dict | None = None) -> str:
    lines = []
    if site.purpose in BANNER:
        lines += [BANNER[site.purpose], ""]
    if report and (report.get("mixed_code_versions") or report.get("dirty_code")):
        lines += [f"> **CODE VERSION WARNING.** Plans come from {report.get('code_versions')} (dirty tree: {report.get('dirty_code')}); "
                  "results are only comparable when produced by the same committed code.", ""]
    if site.gt_kind == "pseudo_lidar":
        lines += ["> **PSEUDO-GT.** Reference values come from our own LiDAR reconstruction, not a laser/tape.",
                  "> These numbers measure consistency between tiers, not accuracy. They are not the benchmark.", ""]
    if site.gt_kind == "synthetic":
        lines += ["> **SYNTHETIC.** Exact geometry of a rendered capture: a pipeline smoke test, not a benchmark.", ""]
    lines += [f"# Evaluation: {site.site_id} (gt_kind={site.gt_kind})", "", "| capture | tier | gate results |", "|---|---|---|"]
    for k, v in gate.items():
        if k in ("repeatability", "head_to_head"): continue
        lines.append(f"| {k} | {v['tier']} | " + ", ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in v.items() if a != "tier") + " |")
    for grp, v in gate.get("repeatability", {}).items():
        lines.append(f"\nRepeatability `{grp}`: pass fraction {v['pass_frac']}")
    if "head_to_head" in gate:
        lines.append(f"\nHead-to-head beat-or-tie fraction: {gate['head_to_head']['beat_or_tie_frac']:.2f}")
    lines += _question_tables(results)
    lines += ["", f"Figure: `{site.site_id}_walls.png` (predicted vs reference wall lengths with 90 % intervals); plans: `<run>/<capture>/plan.png`."]
    return "\n".join(lines) + "\n"


REPO = Path(__file__).resolve().parents[3]


def capture_input(site: Site, cap) -> Path:
    """Capture path relative to the site directory, else to the repository root."""
    for base in (site.root, REPO):
        if base is not None and (Path(base) / cap.path).exists():
            return Path(base) / cap.path
    return REPO / cap.path


def execute(site: Site, run: Path, force: bool = False) -> dict:
    """Run each capture through its tier's production CLI in its own process."""
    log = {}
    for cap in site.captures:
        out = run / cap.id
        src = capture_input(site, cap)
        if (out / "plan.json").exists() and not force:
            log[cap.id] = {"status": "reused"}; continue
        if not src.exists():
            log[cap.id] = {"status": "input_missing", "path": str(src)}; continue
        cmd = [sys.executable, "-m", f"roomscan.{cap.tier}.cli", str(src), str(out), *cap.args]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        log[cap.id] = {"status": "ok" if r.returncode == 0 else "failed", "returncode": r.returncode, "seconds": round(time.time() - t0, 1),
                       "command": " ".join(cmd[1:]), **({"stderr_tail": r.stderr[-1500:]} if r.returncode else {})}
        print(f"[execute] {cap.id}: {log[cap.id]['status']} ({log[cap.id]['seconds']} s)", flush=True)
    return log


def run(site_dir: str, run_dir: str, do_execute: bool = False, force: bool = False) -> dict:
    site = load_site(site_dir); run = Path(run_dir)
    execution = execute(site, run, force) if do_execute else None
    results = []
    for cap in site.captures:
        p = run / cap.id / "plan.json"
        if not p.exists():
            results.append({"capture": cap.id, "tier": cap.tier, "missing_prediction": True, "walls": [], "ceilings": [], "areas": [], "openings": [],
                            "opening_hit_rate": None, "adjacency": {"correct": False}, "max_room_overlap_m2": 0.0, "rooms_matched": 0,
                            "rooms_expected": len(cap.rooms or site.rooms)})
            continue
        plan = Plan.model_validate_json(p.read_text())
        res = evaluate_capture(site, cap, plan)
        res.update({"question": cap.question, "variant": cap.variant, "diagnostics": diagnostics.summarise(cap.tier, run / cap.id / "debug.json"),
                    "provenance": plan.provenance})
        results.append(res)
    gate = gates(site, results)
    residuals = calibration_residuals(site, results)
    out = run / "eval"; out.mkdir(parents=True, exist_ok=True)
    commits = {(r.get("provenance") or {}).get("code_commit") for r in results if not r.get("missing_prediction")}
    dirty = any((r.get("provenance") or {}).get("code_dirty") for r in results)
    report = {"site": site.site_id, "gt_kind": site.gt_kind, "purpose": site.purpose, "pseudo_gt": site.gt_kind == "pseudo_lidar",
              "code_versions": sorted(c for c in commits if c) or None, "mixed_code_versions": len(commits) > 1, "dirty_code": dirty,
              "gates": gate, "captures": results, "calibration_residuals": residuals, "execution": execution}
    (out / f"{site.site_id}.json").write_text(json.dumps(report, indent=1, default=str))
    (out / f"{site.site_id}.md").write_text(render_md(site, results, gate, report))
    try:
        from roomscan.bench.figures import wall_figure
        wall_figure(site, results, out / f"{site.site_id}_walls.png")
    except Exception as e:            # the numbers are the record; a plotting failure must not lose them
        report["figure_error"] = repr(e)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("site"); ap.add_argument("--run", required=True)
    ap.add_argument("--execute", action="store_true", help="run each capture through its tier's production pipeline first")
    ap.add_argument("--force", action="store_true", help="with --execute: re-run captures whose plan.json exists")
    a = ap.parse_args()
    rep = run(a.site, a.run, a.execute, a.force)
    print(render_md(load_site(a.site), rep["captures"], rep["gates"], rep))


if __name__ == "__main__":
    main()

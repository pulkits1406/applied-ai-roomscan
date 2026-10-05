"""Damage regions, concealed-damage rules and scope line items attached to a measured plan.

    uv run roomscan-damage <plan.json> <observations.json> <out_plan.json>

STATUS: the mechanism is implemented and tested; there is NO damage detector and NO damage
benchmark yet (no captures with staged damage exist). Observations come from an external
detector or a manual annotation in this format:

    {"source": "manual" | "<detector name and version>",
     "observations": [{"surface_id": "r1_w2", "class": "water_stain",
                       "region_uv": [[u, v], ...],          # metres in the surface frame (below)
                       "confidence": 0.9}]}

Surface frame: walls u = metres along the wall from its first polygon vertex, v = metres above
the floor; floors and ceilings use the plan (x, y). Each observation becomes a DamageRegion
whose area interval covers a boundary uncertainty of BOUNDARY_SIGMA_M (uncalibrated). Every
concealed-damage flag names the rule that fired; every scope item is keyed to a surface and its
quantity carries an interval derived from the measured surface/region.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from roomscan.assemble import meas
from roomscan.model import ConcealedDamageFlag, DamageRegion, Plan, ScopeItem

CLASSES = ("water_stain", "mould", "crack", "impact_hole", "flood_line")
BOUNDARY_SIGMA_M = 0.02
WET_ROOM_WORDS = ("bath", "shower", "wc", "toilet", "kitchen", "utility", "laundry")
METHOD = "damage_rules_v1_uncal"

# rule id -> (description, predicate(observation, surface, plan) -> bool, surfaces to flag)
RULES_DOC = {
    "CD1": "water stain or mould on a ceiling: a leak above the ceiling can have wetted the cavity, insulation and the tops of the walls",
    "CD2": "water damage, mould or a flood line within 0.3 m of the floor on a wall: moisture may have wicked into the wall base, "
           "skirting and floor below",
    "CD3": "water damage or mould on a wall shared with a wet room (bath/shower/WC/kitchen/utility): the source may be a concealed "
           "pipe or failed seal behind the wall",
    "CD4": "crack longer than 1.0 m on a wall or ceiling: may indicate movement behind the finish",
}
WET = {"water_stain", "mould", "flood_line"}


def _poly_area(p):
    p = np.asarray(p, float)
    return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) - np.dot(p[:, 1], np.roll(p[:, 0], -1))))


def _perimeter(p):
    p = np.asarray(p, float)
    return float(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1).sum())


def _extent(p):
    p = np.asarray(p, float)
    return float(np.max(np.ptp(p, axis=0)))


def _is_wet(r) -> bool:
    return r is not None and any(w in (r.label or "").lower() for w in WET_ROOM_WORDS)


def _shared_with(plan: Plan, s) -> set[str]:
    """Rooms on the other side of wall surface s: its shared_with_room_id, and the other room of
    every opening placed in this wall."""
    out = {s.shared_with_room_id} if s.shared_with_room_id else set()
    for o in plan.openings:
        if o.wall_id == s.id:
            out |= {x for x in o.room_ids if x != s.room_id}
    return out


def attach(plan: Plan, observations: dict) -> Plan:
    surf = {s.id: s for s in plan.surfaces}
    room = {r.id: r for r in plan.rooms}
    src = observations.get("source", "unspecified")
    regions, flags, scope = list(plan.damage), list(plan.concealed_damage_flags), list(plan.scope)
    fired = set()
    for k, o in enumerate(observations.get("observations", [])):
        sid, cls = o["surface_id"], o["class"]
        if sid not in surf:
            raise ValueError(f"observation {k}: unknown surface {sid}")
        if cls not in CLASSES:
            raise ValueError(f"observation {k}: unknown damage class {cls} (known: {CLASSES})")
        s = surf[sid]
        poly = [tuple(map(float, x)) for x in o["region_uv"]]
        a = _poly_area(poly)
        sig = BOUNDARY_SIGMA_M * _perimeter(poly)                 # area change from moving the boundary by sigma
        rid = f"d{len(regions)}"
        regions.append(DamageRegion(id=rid, surface_id=sid, damage_class=cls, region_uv=poly,
                                    area=meas(a, sig, "m2", f"{METHOD}:{src}"), confidence=float(o.get("confidence", 1.0))))
        r = room.get(s.room_id)
        low = min(v for _, v in poly) < 0.3 if s.kind == "wall" else False
        shared_wet = s.kind == "wall" and any(_is_wet(room.get(other)) for other in _shared_with(plan, s))
        checks = [("CD1", s.kind == "ceiling" and cls in WET, [sid] + (list(r.wall_ids) if r else [])),
                  ("CD2", s.kind == "wall" and cls in WET and low, [sid] + [x.id for x in plan.surfaces if x.room_id == s.room_id and x.kind == "floor"]),
                  ("CD3", cls in WET and shared_wet, [sid]),
                  ("CD4", cls == "crack" and _extent(poly) > 1.0, [sid])]
        for rule, hit, sids in checks:
            if hit and (rule, sid) not in fired:
                fired.add((rule, sid))
                flags.append(ConcealedDamageFlag(id=f"c{len(flags)}", rule_id=rule, surface_ids=sids, reason=RULES_DOC[rule]))
        scope += _scope(s, cls, poly, a, sig, len(scope))
    return plan.model_copy(update={"damage": regions, "concealed_damage_flags": flags, "scope": scope})


def _scope(s, cls, poly, a, sig, n0) -> list[ScopeItem]:
    """Line items keyed to the surface. Repairs are quantified on the damaged region (+20 % cut-out
    margin); refinishing is quantified on the whole measured surface, using its own interval."""
    items = []
    if cls in ("water_stain", "mould", "flood_line", "impact_hole"):
        items.append(ScopeItem(id=f"s{n0}", surface_id=s.id, action=f"repair {cls.replace('_', ' ')} (cut out and replace / treat)",
                               quantity=meas(1.2 * a, 1.2 * sig, "m2", METHOD)))
    if cls == "crack":
        L = _extent(poly)
        items.append(ScopeItem(id=f"s{n0 + len(items)}", surface_id=s.id, action="rake out, fill and make good crack",
                               quantity=meas(L, BOUNDARY_SIGMA_M * 2, "m", METHOD)))
    sa = s.area
    if sa.value is not None:
        items.append(ScopeItem(id=f"s{n0 + len(items)}", surface_id=s.id, action=f"stain-block / prime and repaint {s.kind}",
                               quantity=sa.model_copy(update={"method": f"{METHOD}:surface_area"})))
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan"); ap.add_argument("observations"); ap.add_argument("out")
    a = ap.parse_args(argv)
    plan = attach(Plan.model_validate_json(Path(a.plan).read_text()), json.loads(Path(a.observations).read_text()))
    Path(a.out).write_text(plan.model_dump_json(indent=1))
    print(json.dumps({"damage_regions": len(plan.damage), "concealed_flags": [f.rule_id for f in plan.concealed_damage_flags],
                      "scope_items": len(plan.scope)}))


if __name__ == "__main__":
    main()

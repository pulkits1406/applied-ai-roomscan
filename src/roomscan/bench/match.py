"""Correspondence between a predicted Plan and tape/laser ground truth."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from roomscan.bench.gt import GtRoom
from roomscan.model import Plan, Room


def match_rooms(plan: Plan, gt_rooms: list[GtRoom], room_map: dict[str, str]) -> dict[str, str]:
    """Predicted room id -> GT room id. Explicit map and matching labels win; only UNLABELLED rooms
    are then assigned by Hungarian matching on perimeter and wall count (size similarity, not
    identity: the caller reports such matches as unverified). A labelled room whose label is not a
    GT room (e.g. a photo folder of a room the site does not score) is never matched."""
    out = {p: g for p, g in room_map.items()}
    gt_ids = {g.id for g in gt_rooms}
    for r in plan.rooms:
        if r.id not in out and r.label in gt_ids and r.label not in out.values():
            out[r.id] = r.label
    pr = [r for r in plan.rooms if r.id not in out and r.label is None]
    gr = [g for g in gt_rooms if g.id not in out.values()]
    if pr and gr:
        C = np.zeros((len(pr), len(gr)))
        for i, r in enumerate(pr):
            for j, g in enumerate(gr):
                known = [w.length for w in g.walls if w.length is not None]
                if len(known) == len(g.walls):
                    gp = sum(known)
                    C[i, j] = abs(r.perimeter.value - gp) / gp + 0.1 * abs(len(r.wall_ids) - len(g.walls))
                elif known and len(r.wall_ids) == len(g.walls):
                    d = {}
                    match_walls(plan, r, g, d)                  # partial tape GT: fit of the measured walls only
                    C[i, j] = d["best_cost_m"] / sum(known)
                else:
                    C[i, j] = 1e3
        for i, j in zip(*linear_sum_assignment(C)):
            if C[i, j] < 0.35:
                out[pr[i].id] = gr[j].id
    return out


def match_walls(plan: Plan, room: Room, gt: GtRoom, detail: dict | None = None) -> dict[str, str]:
    """GT wall id -> predicted wall surface id via the cyclic shift / reflection of the predicted
    wall sequence that minimises the length error over the MEASURED GT walls (partial tape GT
    leaves some lengths None). A different wall count is a topology mismatch: no walls are
    matched; the caller reports the room as a topology failure. If `detail` is given it receives
    the best and second-best costs: an alternative within 20 % of the best means the wall
    correspondence (and therefore the per-wall errors) is ambiguous."""
    surf = {s.id: s for s in plan.surfaces}
    pl = [surf[w].length.value for w in room.wall_ids]
    gl = [w.length for w in gt.walls]
    n, m = len(pl), len(gl)
    if n != m:
        return {}
    cands = []
    for direction in (1, -1):
        seq = room.wall_ids[::direction]
        lens = pl[::direction]
        for shift in range(n):
            pairs = [(gt.walls[k].id, seq[(k + shift) % n]) for k in range(n)]
            cost = sum(abs(gl[k] - lens[(k + shift) % n]) for k in range(n) if gl[k] is not None)
            cands.append((cost, pairs))
    cands.sort(key=lambda c: c[0])
    best = cands[0]
    if detail is not None:
        alt = [c[0] for c in cands[1:] if {p for p in c[1] if gl[[w.id for w in gt.walls].index(p[0])] is not None}
               != {p for p in best[1] if gl[[w.id for w in gt.walls].index(p[0])] is not None}]
        detail.update({"best_cost_m": round(best[0], 4), "second_cost_m": round(alt[0], 4) if alt else None,
                       "ambiguous": bool(alt and alt[0] <= 1.2 * best[0] + 0.01)})
    return dict(best[1])


def match_openings(plan: Plan, room: Room, gt: GtRoom, wall_map: dict[str, str], max_width_err: float = 0.15):
    """Returns (pairs [(gt_opening, pred_opening)], missed gt ids, phantom pred ids).
    Cost = width error, with a penalty when the opening sits on a different wall than GT says."""
    preds = [o for o in plan.openings if o.id in room.opening_ids]
    gts = gt.openings
    if not gts or not preds:
        return [], [g.id for g in gts], [p.id for p in preds]
    C = np.full((len(gts), len(preds)), 1e3)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if p.width.value is None:
                continue
            kind_ok = g.kind == p.kind or p.kind == "unknown" or {g.kind, p.kind} == {"door", "doorway"}
            if not kind_ok:
                continue
            wall_pen = 0.0 if wall_map.get(g.wall) == p.wall_id else 0.05
            C[i, j] = abs(g.width - p.width.value) + wall_pen
    rows, cols = linear_sum_assignment(C)
    pairs = [(gts[i], preds[j]) for i, j in zip(rows, cols) if C[i, j] <= max_width_err]
    mg = {g.id for g, _ in pairs}
    mp = {p.id for _, p in pairs}
    return pairs, [g.id for g in gts if g.id not in mg], [p.id for p in preds if p.id not in mp]

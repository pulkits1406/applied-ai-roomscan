"""Photo tier: per-room folders of stills -> Plan (schema 0.1), rooms unplaced.

Per room: load photos (EXIF focal only) -> MoGe-2 + MapAnything -> ensemble-scaled,
gravity-aligned cloud (reconstruct) -> segmentation / walls / levels shared with LiDAR (cloudroom)
-> RoomInput with photo-tier sigmas -> assemble. Whole-property stitching from photo folders is
not solved (e17: 0/36 doorway links), so every room is emitted with placed=False and laid out side
by side for display only.

Provisional, UNCALIBRATED error model (1 sigma):
  scale    SCALE_SIGMA_REL (8.7 %) per room, 'shared:scale' (e21 fixed +-14.3 % at 90 %)
  faces    FACE_SIGMA by geometry mode for observed faces, FACE_SIGMA_INFERRED for inferred faces
  ceiling  LEVEL_SIGMA per level, plus scale
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from roomscan import geometry as G
from roomscan.assemble import RoomInput, assemble
from roomscan.cloudroom import room_from_cloud
from roomscan.model import CaptureInfo, Plan
from roomscan.photo import inference, reconstruct as RC
from roomscan.photo.images import load_photo, room_folders

FACE_SIGMA = {"moge": 0.03, "ma": 0.06}
FACE_SIGMA_INFERRED = 0.15
LEVEL_SIGMA = 0.03
LAYOUT_GAP = 1.0
METHOD = "photo_moge_mapanything_ensemble_uncal"


def _plan_xy(poly_uv):
    """Room frame (u, v) -> display plan coordinates (x, y) = (u, -v), as the LiDAR tier maps (x, z)."""
    return np.stack([poly_uv[:, 0], -poly_uv[:, 1]], 1)


def build(folder: str | Path, cache: str | Path, mode: str = "moge") -> tuple[Plan, dict]:
    folders = room_folders(folder)
    cache = Path(cache)
    debug, rooms, flags, x0 = {"mode": mode, "rooms": {}}, [], [], 0.0
    if not folders:
        flags.append("no_room_folders: expected one sub-folder of photos per room")
    sets = {name: [load_photo(p) for p in paths] for name, paths in folders.items()}
    model = inference.MoGe() if sets else None       # one MoGe load for all rooms, released before MapAnything
    mos = {name: inference.moge(ph, cache / name, model) for name, ph in sets.items()}
    if model is not None:
        model.close()
    mas = dict(zip(sets, inference.mapanything_many(list(sets.values()), [cache / n for n in sets]))) if sets else {}
    for name, photos in sets.items():
        no_focal = [p.path.name for p in photos if p.f_px is None]
        if no_focal:
            flags.append(f"{name}_no_exif_focal: {len(no_focal)} photo(s) without a 35 mm-equivalent focal; field of view estimated by the model")
        if len(photos) < 2:
            flags.append(f"{name}_single_photo: one photo cannot show every wall; walls outside it are inferred or missing")
        mo, ma = mos[name], mas[name]
        if ma is None:
            flags.append(f"{name}_reconstruction_failed: the multi-view model produced no output")
            continue
        pc = RC.room_cloud(mo, ma, mode)
        g = room_from_cloud(pc.P, pc.N, pc.cams)
        rec = {"photos": len(photos), "scale_ma_to_m": round(pc.scale_ma_to_m, 4), "r_moge_over_ma": round(pc.r_moge_over_ma, 4),
               "log_scale_disagreement": round(pc.scale_disagreement, 3), "n_points": int(len(pc.P))}
        if isinstance(g, str):
            debug["rooms"][name] = {**rec, "result": g}
            flags.append(f"{name}_{g}: photos reconstructed but no room outline could be built")
            continue
        src = g.walls.source
        observed = [s in ("close", "all") for s in src]
        poly = _plan_xy(g.walls.poly_uv)
        poly = poly - poly.min(0) + np.array([x0, 0.0])
        x0 = float(poly[:, 0].max()) + LAYOUT_GAP
        ch = g.ceil_h
        rooms.append(RoomInput(
            id=name, label=name, polygon=poly,
            face_sigma=[FACE_SIGMA[mode] if o else FACE_SIGMA_INFERRED for o in observed],
            face_status=["measured" if o else "inferred" for o in observed],
            scale_sigma_rel=RC.SCALE_SIGMA_REL, ceiling_h=ch,
            ceiling_sigma=0.0 if ch is None else float(np.hypot(np.sqrt(2) * LEVEL_SIGMA, RC.SCALE_SIGMA_REL * ch)),
            method=f"{METHOD}_{mode}", placed=False))
        debug["rooms"][name] = {**rec, "result": "ok", "step": g.step, "n_walls": len(poly), "wall_sources": src,
                                "ceiling_points": g.n_ceil, "area_m2": round(abs(G.polygon_area(poly)), 3)}
        if g.step == "inferred" or not all(observed):
            flags.append(f"{name}_walls_inferred: {observed.count(False)} wall(s) not seen at structural height in the photos; "
                         f"placed at the reconstructed floor boundary (face sigma {FACE_SIGMA_INFERRED} m)")
        if ch is None:
            flags.append(f"{name}_ceiling_not_observed: the photos show too little ceiling")
    if rooms:
        flags.append("rooms_unplaced: photo folders are not stitched into one plan (no validated method: e17 doorway photos 0/36); "
                     "room positions in this plan carry no information")
    plan = assemble("photo", rooms, [], CaptureInfo(tier="photo", app="stock Camera (stills)", inputs=[str(folder)]),
                    f"{METHOD}_{mode}", flags, "none: rooms measured independently, not placed")
    return plan, debug

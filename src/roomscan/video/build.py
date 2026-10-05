"""Video tier v0: plain clip -> overlapping keyframe windows -> per-window rooms -> Plan (unplaced).

Evidence (e20, e25): no RGB-only tracker gave geometry coherent enough to measure a room over a
10-60 s piece of the supplied walkthrough (PnP-on-MoGe pieces drift ~25 cm per 20 s; COLMAP-SIFT
pieces fragment and their self-estimated scale is off by 12-30 %), while 8 overlapping frames
reconstructed jointly by MapAnything gave a coherent room (e23). So the clip is cut into short
windows, each measured like a photo set (MoGe-2 + MapAnything ensemble scale, shared
segmentation/walls/levels), and each window is accepted only if its reconstruction is coherent
(GT-free check: the camera heights above the reconstructed floor agree). Windows are not linked:
every accepted room is emitted unplaced and labelled with its time span; consecutive windows that
measure the same room are merged, but a room revisited later can appear twice (flagged).

Intrinsics are not read from the file: the field of view is estimated by MoGe-2 on sampled frames
(median), and that uncertainty is added to the scale term.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roomscan import geometry as G
from roomscan.assemble import RoomInput, assemble
from roomscan.cloudroom import face_terms, floor_level, room_from_cloud, stability
from roomscan.model import CaptureInfo, Plan
from roomscan.photo import inference, reconstruct as RC
from roomscan.photo.images import Photo
from roomscan.video.frames import read

WINDOW_S = 12.0
FRAMES_PER_WINDOW = 8
FOV_SAMPLES = 16
MAX_CAM_HEIGHT_SPREAD = 0.15     # m, std of camera heights above the floor for a coherent window
MIN_CAM_HEIGHT = 0.5
SAME_ROOM_REL = 0.15             # consecutive windows whose sorted extents agree within this are one room
FACE_SIGMA, FACE_SIGMA_INFERRED, LEVEL_SIGMA = 0.03, 0.15, 0.03
SIG_INTRINSICS_REL = 0.03        # provisional: focal from MoGe on the clip instead of EXIF
METHOD = "video_windows_moge_mapanything_uncal"


def _photos(kfs, d: Path, f_px):
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for k in kfs:
        p = d / f"kf_{k.index:05d}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), cv2.cvtColor(k.rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append(Photo(p, k.rgb, f_px))
    return out


def estimate_focal(kfs, cache: Path, model=None) -> float:
    idx = np.linspace(0, len(kfs) - 1, min(FOV_SAMPLES, len(kfs))).round().astype(int)
    ph = _photos([kfs[i] for i in idx], cache / "frames", None)
    mo = inference.moge(ph, cache / "fov", model)
    w = kfs[0].rgb.shape[1]
    return float(np.median([m["K_norm"][0, 0] * w for m in mo]))


def windows(kfs):
    t = np.array([k.t for k in kfs])
    out, t0 = [], t[0]
    while t0 < t[-1]:
        idx = np.nonzero((t >= t0) & (t < t0 + WINDOW_S))[0]
        if len(idx) >= 4:
            out.append([kfs[i] for i in idx[np.linspace(0, len(idx) - 1, min(FRAMES_PER_WINDOW, len(idx))).round().astype(int)]])
        t0 += WINDOW_S
    return out


def build(video: str | Path, cache: str | Path, rotate="auto", fps: float = 2.0) -> tuple[Plan, dict]:
    cache = Path(cache)
    kfs, info = read(video, fps, rotate)
    model = inference.MoGe()                  # one MoGe load for the whole clip, released before MapAnything starts
    f_px = estimate_focal(kfs, cache, model)
    debug = {"video": info, "focal_px_estimated": round(f_px, 1), "fov_x_deg": round(float(np.degrees(2 * np.arctan(kfs[0].rgb.shape[1] / (2 * f_px)))), 1),
             "windows": []}
    wins = windows(kfs)
    sets = [_photos(w, cache / "frames", f_px) for w in wins]
    mos = [inference.moge(s, cache / f"w{i:03d}", model) for i, s in enumerate(sets)]
    model.close()
    mas = inference.mapanything_many(sets, [cache / f"w{i:03d}" for i in range(len(sets))])
    cands = []
    for i, (w, mo, ma) in enumerate(zip(wins, mos, mas)):
        rec = {"window": i, "t": [round(w[0].t, 1), round(w[-1].t, 1)]}
        if ma is None:
            debug["windows"].append({**rec, "result": "mapanything_failed"}); continue
        pc = RC.room_cloud(mo, ma, "moge")
        fl = floor_level(pc.P, pc.N, pc.cams)
        hts = None if fl is None else pc.cams[:, 1] - fl
        rec["cam_heights"] = None if hts is None else np.round(hts, 2).tolist()
        if hts is None or np.std(hts) > MAX_CAM_HEIGHT_SPREAD or hts.min() < MIN_CAM_HEIGHT:
            debug["windows"].append({**rec, "result": "incoherent" if hts is not None else "no_floor"}); continue
        g = room_from_cloud(pc.P, pc.N, pc.cams)
        if isinstance(g, str):
            debug["windows"].append({**rec, "result": g}); continue
        st = stability(pc.P, pc.N, pc.cams, g)
        rec.update({"stability_sigma_m": [round(x, 4) for x in st.sigma_m], "stability_runs_without_outline": st.runs_without_outline})
        ext = np.sort(np.ptp(g.walls.poly_uv, 0))
        measured = sum(s in ("close", "all") for s in g.walls.source)
        rec.update({"result": "ok", "n_walls": len(g.walls.poly_uv), "measured_faces": measured, "extent_m": np.round(ext, 3).tolist(),
                    "area_m2": round(abs(G.polygon_area(g.walls.poly_uv)), 3), "scale_ma_to_m": round(pc.scale_ma_to_m, 4)})
        debug["windows"].append(rec)
        if cands and cands[-1]["window"] == i - 1 and np.all(np.abs(np.log(ext / cands[-1]["ext"])) < SAME_ROOM_REL):
            prev = cands[-1]
            prev["merged"].append(i)
            if measured > prev["measured"]:
                prev.update({"g": g, "ext": ext, "measured": measured, "t1": w[-1].t, "window": i, "st": st, "h_spread": float(np.std(hts))})
            else:
                prev.update({"t1": w[-1].t, "window": i})
            continue
        cands.append({"g": g, "ext": ext, "measured": measured, "t0": w[0].t, "t1": w[-1].t, "window": i, "merged": [i],
                      "st": st, "h_spread": float(np.std(hts))})
    rooms, flags, x0 = [], [], 0.0
    scale_rel = float(np.hypot(RC.SCALE_SIGMA_REL, SIG_INTRINSICS_REL))
    for k, c in enumerate(cands):
        g = c["g"]
        observed = [s in ("close", "all") for s in g.walls.source]
        poly = np.stack([g.walls.poly_uv[:, 0], -g.walls.poly_uv[:, 1]], 1)
        poly = poly - poly.min(0) + np.array([x0, 0.0])
        x0 = float(poly[:, 0].max()) + 1.0
        ch = g.ceil_h
        rid = f"v{k}"
        rooms.append(RoomInput(
            id=rid, label=None, polygon=poly,
            face_sigma=[FACE_SIGMA if o else FACE_SIGMA_INFERRED for o in observed], face_status=["measured" if o else "inferred" for o in observed],
            face_terms=face_terms(g.walls.source, c["st"], FACE_SIGMA, FACE_SIGMA_INFERRED, c["h_spread"]),
            scale_sigma_rel=scale_rel, ceiling_h=ch,
            ceiling_sigma=0.0 if ch is None else float(np.hypot(np.sqrt(2) * LEVEL_SIGMA, scale_rel * ch)), method=METHOD, placed=False))
        flags.append(f"{rid}_time_span: measured from {c['t0']:.1f}-{c['t1']:.1f} s of the clip (windows {c['merged']})")
        if c["st"].unstable_topology:
            flags.append(f"{rid}_topology_unstable: the room outline changes under small irrelevant perturbations; intervals widened")
        if not all(observed):
            flags.append(f"{rid}_walls_inferred: {observed.count(False)} wall(s) not seen at structural height in that window")
    n_ok = sum(1 for w in debug["windows"] if w["result"] == "ok")
    flags.append(f"video_fragmented: {len(wins)} windows of {WINDOW_S:.0f} s, {n_ok} coherent, {len(rooms)} room(s) after merging consecutive "
                 "windows; rooms are not linked or placed (no continuous accurate tracking on RGB alone, e20/e25); a revisited room can appear "
                 "more than once and rooms without a coherent window are missing")
    flags.append(f"intrinsics_estimated: no focal length in the video file; field of view {debug['fov_x_deg']} deg estimated from the frames")
    plan = assemble("video", rooms, [], CaptureInfo(tier="video", inputs=[str(video)]), METHOD, flags,
                    "none: windows measured independently, not placed")
    return plan, debug

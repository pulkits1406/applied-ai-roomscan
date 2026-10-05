"""Output representation shared by all tiers (the published JSON contract).

Frame: property frame is right-handed, metres, z up (gravity); plan coordinates are (x, y).
Every numeric quantity reported to a user is a Measurement and must carry an interval; a quantity
that could not be observed is reported with status "not_observed" and value None rather than
guessed. Damage, concealed-damage flags and scope items reference surfaces by id, so they attach
to stable geometry entities produced by the measurement pipeline.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

Tier = Literal["photo", "video", "lidar"]


class Interval(BaseModel):
    lo: float
    hi: float
    level: float = Field(0.9, gt=0, lt=1, description="nominal coverage probability")
    calibrated: bool = Field(False, description="True only when the width comes from an empirical calibration against laser/tape GT")
    calibration_ref: Optional[str] = Field(None, description="id of the calibration run the width came from; required when calibrated")

    @model_validator(mode="after")
    def _ordered(self):
        if self.lo > self.hi:
            raise ValueError("interval lo > hi")
        if self.calibrated and not self.calibration_ref:
            raise ValueError("a calibrated interval must name its calibration run")
        return self


class Measurement(BaseModel):
    value: Optional[float]
    unit: Literal["m", "m2", "deg", "count"]
    interval: Optional[Interval]
    status: Literal["measured", "inferred", "not_observed"] = "measured"
    method: str = Field(..., description="how the value was produced, e.g. 'plane_fit_lidar'")
    error_budget: dict[str, float] = Field(default_factory=dict, description=(
        "1-sigma contributions by source, same unit. Keys prefixed 'shared:' are fully correlated across all "
        "measurements of the same room (e.g. 'shared:scale' at the photo tier) and must not be combined as independent."))

    @model_validator(mode="after")
    def _interval_required(self):
        if self.status != "not_observed":
            if self.value is None or self.interval is None:
                raise ValueError("observed measurements need a value and an interval")
            if not (self.interval.lo <= self.value <= self.interval.hi):
                raise ValueError("value outside its interval")
        return self


class Plane(BaseModel):
    normal: tuple[float, float, float]
    offset: float  # n . p + offset = 0


class Surface(BaseModel):
    id: str
    kind: Literal["floor", "ceiling", "wall"]
    room_id: str
    plane: Plane
    polygon: list[tuple[float, float, float]] = Field(..., description="boundary in the property frame")
    area: Measurement
    length: Optional[Measurement] = None  # walls: horizontal extent
    height: Optional[Measurement] = None  # walls: floor-to-ceiling extent
    shared_with_room_id: Optional[str] = None


class Opening(BaseModel):
    id: str
    kind: Literal["door", "window", "doorway", "unknown"]
    wall_id: str
    room_ids: list[str]
    width: Measurement = Field(..., description="clear opening between finished jambs")
    height: Optional[Measurement] = None
    sill_height: Optional[Measurement] = None
    offset_along_wall: Optional[Measurement] = None
    detection_confidence: float = Field(..., ge=0, le=1)


class Room(BaseModel):
    id: str
    label: Optional[str] = None
    polygon: list[tuple[float, float]] = Field(..., description="floor outline, plan coordinates, counter-clockwise")
    floor_area: Measurement
    perimeter: Measurement
    ceiling_height: Measurement
    wall_ids: list[str]
    opening_ids: list[str] = Field(default_factory=list)
    placement_sigma_m: Optional[float] = Field(None, description="1-sigma uncertainty of this room's placement in the stitched plan")
    placed: bool = Field(True, description=(
        "False when the room could not be placed in a common frame (e.g. photo folders not stitched): its polygon is in the "
        "room's own frame, laid out for display only, and its position carries no information"))


class Adjacency(BaseModel):
    room_a: str
    room_b: str
    via_opening_id: Optional[str] = None
    shared_wall_ids: list[str] = Field(default_factory=list)


class DamageRegion(BaseModel):
    id: str
    surface_id: str
    damage_class: str
    region_uv: list[tuple[float, float]] = Field(..., description="polygon in the surface's own 2-D frame, metres")
    area: Measurement
    confidence: float = Field(..., ge=0, le=1)


class ConcealedDamageFlag(BaseModel):
    id: str
    rule_id: str
    surface_ids: list[str]
    reason: str


class ScopeItem(BaseModel):
    id: str
    surface_id: str
    action: str
    quantity: Measurement


class CaptureInfo(BaseModel):
    tier: Tier
    device: Optional[str] = None
    app: Optional[str] = None
    app_version: Optional[str] = None
    inputs: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    schema_version: Literal["0.1"] = "0.1"
    capture: CaptureInfo
    rooms: list[Room]
    surfaces: list[Surface]
    openings: list[Opening]
    adjacency: list[Adjacency]
    footprint_area: Measurement
    damage: list[DamageRegion] = Field(default_factory=list)
    concealed_damage_flags: list[ConcealedDamageFlag] = Field(default_factory=list)
    scope: list[ScopeItem] = Field(default_factory=list)
    drift_correction: Optional[str] = Field(None, description="method applied to poses before stitching")
    provenance: Optional[dict] = Field(None, description=(
        "code version that produced the plan (git commit, dirty flag, library versions) and pipeline parameters; "
        "results are comparable only when produced by the same code"))
    quality_flags: list[str] = Field(default_factory=list, description=(
        "capture-quality problems that limit the result, each 'code: explanation' (e.g. upper walls not "
        "observed, room without a polygon); an empty or partial plan must say why"))

    @model_validator(mode="after")
    def _references_resolve(self):
        rooms = {r.id for r in self.rooms}
        surfaces = {s.id for s in self.surfaces}
        openings = {o.id for o in self.openings}
        for s in self.surfaces:
            if s.room_id not in rooms:
                raise ValueError(f"surface {s.id} references unknown room {s.room_id}")
        for o in self.openings:
            if o.wall_id not in surfaces or not set(o.room_ids) <= rooms:
                raise ValueError(f"opening {o.id} has dangling references")
        for r in self.rooms:
            if not set(r.wall_ids) <= surfaces or not set(r.opening_ids) <= openings:
                raise ValueError(f"room {r.id} has dangling references")
        for a in self.adjacency:
            if {a.room_a, a.room_b} - rooms or (a.via_opening_id and a.via_opening_id not in openings):
                raise ValueError("adjacency has dangling references")
        for d in [*self.damage, *self.scope]:
            if d.surface_id not in surfaces:
                raise ValueError(f"{d.id} references unknown surface {d.surface_id}")
        return self

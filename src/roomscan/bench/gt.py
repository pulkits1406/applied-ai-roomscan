"""Benchmark site format (see benchmark/README.md) and loader.

Tape/laser ground truth records lengths, not coordinates, so walls are listed per room in cyclic
order (clockwise seen from above, starting at the wall holding the entry opening). Repeated
readings are stored raw; the median is the reference value.
"""
from __future__ import annotations

from pathlib import Path
from statistics import median
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

GtKind = Literal["laser", "tape", "pseudo_lidar", "synthetic"]


def _med(v: float | list[float]) -> float:
    return float(median(v)) if isinstance(v, list) else float(v)


class GtWall(BaseModel):
    id: str
    length_m: float | list[float]

    @property
    def length(self) -> float:
        return _med(self.length_m)


class GtOpening(BaseModel):
    id: str
    kind: Literal["door", "window", "doorway"]
    wall: str
    width_m: float | list[float]
    height_m: Optional[float | list[float]] = None
    connects: Optional[str] = None

    @property
    def width(self) -> float:
        return _med(self.width_m)


class GtDamage(BaseModel):
    id: str
    surface: str
    damage_class: str = Field(alias="class")
    extent_m: list[float]


class GtRoom(BaseModel):
    id: str
    walls: list[GtWall]
    ceiling_height_m: Optional[list[float]] = None
    floor_area_m2: Optional[float] = None
    openings: list[GtOpening] = Field(default_factory=list)
    damage: list[GtDamage] = Field(default_factory=list)

    @property
    def ceiling_height(self) -> Optional[float]:
        return _med(self.ceiling_height_m) if self.ceiling_height_m else None

    @property
    def area(self) -> Optional[float]:
        """Explicit area, else the rectangle area when there are exactly four walls."""
        if self.floor_area_m2 is not None:
            return self.floor_area_m2
        if len(self.walls) == 4:
            w = [x.length for x in self.walls]
            return 0.5 * (w[0] + w[2]) * 0.5 * (w[1] + w[3])
        return None


class CaptureSpec(BaseModel):
    id: str
    tier: Literal["photo", "video", "lidar"]
    path: str
    device: Optional[str] = None
    app: Optional[str] = None
    rooms: list[str] = Field(default_factory=list)
    repeat_group: Optional[str] = None
    room_map: dict[str, str] = Field(default_factory=dict, description="predicted room id -> GT room id override")


class IncumbentRoom(BaseModel):
    walls: dict[str, float] = Field(default_factory=dict)
    ceiling_height: Optional[float] = None
    openings: dict[str, float] = Field(default_factory=dict)


class Incumbent(BaseModel):
    app: str
    version: str
    export: str
    rooms: dict[str, IncumbentRoom]


class Site(BaseModel):
    site_id: str
    gt_kind: GtKind
    instrument: Optional[str] = None
    opening_width_definition: str = "clear opening between finished jambs"
    rooms: list[GtRoom]
    captures: list[CaptureSpec]
    incumbent: list[Incumbent] = Field(default_factory=list)
    root: Optional[Path] = None

    @field_validator("rooms")
    @classmethod
    def _unique_rooms(cls, v):
        ids = [r.id for r in v]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate room ids")
        return v

    def room(self, rid: str) -> GtRoom:
        return next(r for r in self.rooms if r.id == rid)


def load_site(site_dir: str | Path) -> Site:
    site_dir = Path(site_dir)
    data = yaml.safe_load((site_dir / "site.yaml").read_text())
    s = Site.model_validate(data)
    s.root = site_dir
    return s

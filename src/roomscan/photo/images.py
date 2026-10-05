"""Photo-tier input: one folder per room, 2-8 stills each, nothing but the image files.

The only camera information used is what an iPhone writes into every photo: EXIF orientation and
the 35 mm-equivalent focal length (an integer, so it carries up to ~2 % rounding error). The
equivalent focal is defined on the 43.27 mm full-frame diagonal, so f_px = f35 / 43.27 * diag_px.
A photo without a usable focal is still accepted; its intrinsics are then estimated by MoGe-2 and
the room is flagged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image, ImageOps

FULL_FRAME_DIAG_MM = 43.27
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
EXIF_IFD = 0x8769


@dataclass
class Photo:
    path: Path
    rgb: np.ndarray            # upright HxWx3 uint8
    f_px: float | None         # focal length in pixels of `rgb`, from EXIF; None if absent

    @property
    def K(self) -> np.ndarray | None:
        if self.f_px is None:
            return None
        h, w = self.rgb.shape[:2]
        return np.array([[self.f_px, 0, (w - 1) / 2], [0, self.f_px, (h - 1) / 2], [0, 0, 1.0]])

    @property
    def fov_x_deg(self) -> float | None:
        return None if self.f_px is None else float(np.degrees(2 * np.arctan(self.rgb.shape[1] / (2 * self.f_px))))


def _open(path: Path) -> Image.Image:
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            import pillow_heif  # optional dependency for original iPhone files
            pillow_heif.register_heif_opener()
        except ImportError as e:
            raise RuntimeError(f"{path.name}: HEIC needs pillow-heif (uv add pillow-heif) or export as JPEG") from e
    return Image.open(path)


def load_photo(path: str | Path, max_side: int = 1024) -> Photo:
    path = Path(path)
    im = _open(path)
    exif = im.getexif()
    f35 = exif.get_ifd(EXIF_IFD).get(ExifTags.Base.FocalLengthIn35mmFilm)
    im = ImageOps.exif_transpose(im).convert("RGB")       # upright as the user saw it
    s = min(1.0, max_side / max(im.size))
    if s < 1.0:
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    w, h = im.size
    f_px = float(f35) / FULL_FRAME_DIAG_MM * float(np.hypot(w, h)) if f35 else None
    return Photo(path, np.asarray(im), f_px)


def room_folders(root: str | Path) -> dict[str, list[Path]]:
    """{room name: sorted image paths} for every sub-folder holding at least one image."""
    out = {}
    for d in sorted(Path(root).iterdir()):
        if d.is_dir():
            imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)
            if imgs:
                out[d.name] = imgs
    return out


def write_photo(path: str | Path, rgb: np.ndarray, f_px: float, make="Apple", model="simulated"):
    """JPEG with the EXIF fields an iPhone photo carries (used to build simulated photo folders)."""
    im = Image.fromarray(rgb)
    exif = Image.Exif()
    exif[ExifTags.Base.Make] = make
    exif[ExifTags.Base.Model] = model
    exif.get_ifd(EXIF_IFD)[ExifTags.Base.FocalLengthIn35mmFilm] = int(round(f_px * FULL_FRAME_DIAG_MM / np.hypot(*rgb.shape[:2])))
    im.save(path, quality=92, exif=exif)

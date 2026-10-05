"""Which code produced a result. Two "uncached runs" that differed in phase 4 had in fact run
different code versions (a fix landed while one process was alive); every plan now records the
code version so results can only be compared when they come from the same code."""
from __future__ import annotations

import platform
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIBS = ("numpy", "scipy", "open3d", "opencv-python", "torch", "moge", "pydantic")


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


@lru_cache(maxsize=1)
def provenance() -> dict:
    """Captured once per process, i.e. for the code the process actually imported."""
    status = _git("status", "--porcelain", "--untracked-files=no", "--", "src", "pyproject.toml", "uv.lock")
    libs = {}
    for name in LIBS:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"code_commit": _git("rev-parse", "HEAD"), "code_dirty": bool(status) if status is not None else None,
            "python": platform.python_version(), "libs": libs}

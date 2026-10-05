"""Isolated interpreters for models whose dependencies cannot share the project venv.

MapAnything pins opencv-headless 4.10 (the project uses opencv 5), and pycolmap cannot share a
process with torch (both bundle libomp: OMP Error #15). Such steps run as subprocesses with the
interpreter resolved here (first hit wins): $ROOMSCAN_PY_<NAME> -> <repo>/.envs/<name>/bin/python
(built by scripts/setup_envs.sh from envs/<name>/) -> the legacy experiment venv.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = {"mapanything": ROOT / "cache/e13/.venv/bin/python", "video": ROOT / "cache/e20/.venv/bin/python"}


def python_for(name: str) -> Path:
    env = os.environ.get(f"ROOMSCAN_PY_{name.upper()}")
    for cand in ([Path(env)] if env else []) + [Path(os.environ.get("ROOMSCAN_ENVS", ROOT / ".envs")) / name / "bin/python", LEGACY.get(name)]:
        if cand is not None and cand.exists():
            return cand
    raise FileNotFoundError(f"no interpreter for '{name}': run scripts/setup_envs.sh {name} (see docs/reproduction.md)")

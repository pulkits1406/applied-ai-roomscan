"""Same raw input -> materially identical plan (bitwise identity is not expected: Open3D's
multithreaded reductions change summation order, ~1e-11 m on the real capture, e26)."""
import json
import os
import subprocess
import sys

import pytest

from roomscan import synthetic
from roomscan.bench.compare import compare, materially_identical
from roomscan.lidar.build import build


def test_lidar_pipeline_is_repeatable(tmp_path):
    cap = synthetic.write(tmp_path / "syn", fps=10)
    a, _ = build(cap)
    b, _ = build(cap)
    assert materially_identical(a, b, tol=1e-6), compare(a, b)
    assert a.provenance and a.provenance["code_commit"]


GPU_SNIPPET = """
import sys, numpy as np
from pathlib import Path
from roomscan.photo.images import Photo
from roomscan.photo.inference import MoGe
rgb = (np.random.default_rng(0).random((480, 360, 3)) * 255).astype(np.uint8)
o = MoGe()(Photo(Path("x.jpg"), rgb, 400.0))
np.save(sys.argv[1], np.nan_to_num(o["points"], nan=-1.0))
"""


@pytest.mark.skipif(os.environ.get("ROOMSCAN_GPU_TESTS") != "1", reason="loads MoGe-2 twice; set ROOMSCAN_GPU_TESTS=1")
def test_moge_is_bitwise_repeatable_across_processes(tmp_path):
    outs = []
    for k in range(2):
        f = tmp_path / f"o{k}.npy"
        subprocess.run([sys.executable, "-c", GPU_SNIPPET, str(f)], check=True, capture_output=True)
        outs.append(f.read_bytes())
    assert outs[0] == outs[1]

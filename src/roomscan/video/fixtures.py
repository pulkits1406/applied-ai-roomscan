"""Test clips with the orientation signalling an iPhone .MOV uses: the track header (tkhd)
transformation matrix (MOV/MP4 store rotation there; FFmpeg >= 7 no longer writes the legacy
'rotate' tag, so the matrix is patched in after encoding)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

def _matrix(deg: int, w: int, h: int) -> bytes:
    """tkhd matrix (a, b, u, c, d, v, x, y, w): a-d, x, y in 16.16 fixed point, u, v, w in 2.30."""
    a, b, c, d = {0: (1, 0, 0, 1), 90: (0, 1, -1, 0), 180: (-1, 0, 0, -1), 270: (0, -1, 1, 0)}[deg]
    tx, ty = {0: (0, 0), 90: (h, 0), 180: (w, h), 270: (0, w)}[deg]
    fx = lambda v: struct.pack(">i", int(v * 65536))                     # noqa: E731
    return fx(a) + fx(b) + struct.pack(">i", 0) + fx(c) + fx(d) + struct.pack(">i", 0) + fx(tx) + fx(ty) + struct.pack(">i", 1 << 30)


def write_clip(path: str | Path, frames: list[np.ndarray], fps: int = 30, rotation: int = 0) -> Path:
    """H.264 .mov of RGB frames whose track header says 'display rotated by `rotation` degrees'."""
    import av
    path = Path(path)
    h, w = frames[0].shape[:2]
    out = av.open(str(path), "w", format="mov")
    st = out.add_stream("h264", rate=fps)
    st.width, st.height, st.pix_fmt = w, h, "yuv420p"
    for f in frames:
        for p in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    if rotation:
        data = bytearray(path.read_bytes())
        i = data.find(b"tkhd")
        version = data[i + 4]
        off = i + 4 + (52 if version == 1 else 40)
        data[off:off + 36] = _matrix(rotation, w, h)
        path.write_bytes(bytes(data))
    return path

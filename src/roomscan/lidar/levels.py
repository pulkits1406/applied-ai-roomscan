"""Floor and ceiling levels of one room from horizontal surfaces inside its polygon.

Only points with near-vertical normals are used (e18: unoriented height histograms picked furniture
tops or a neighbouring ceiling level, up to 110 mm off). The dominant ceiling level is reported;
bulkheads/soffits are a known limitation.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from shapely import contains_xy
from shapely.geometry import Polygon

MIN_CEILING_POINTS = 300


def levels(P, N, poly_uv, Rm, floor_guess):
    """Returns (floor_y, ceiling_y or None, ceiling_point_count)."""
    shp = Polygon(poly_uv).buffer(-0.2)
    if shp.is_empty:
        shp = Polygon(poly_uv)
    uv = P[:, [0, 2]] @ Rm.T
    m = contains_xy(shp, uv[:, 0], uv[:, 1]) & (np.abs(N[:, 1]) > 0.9)
    y = P[m, 1]
    fl = y[np.abs(y - floor_guess) < 0.08]
    floor = float(np.median(fl)) if len(fl) > 200 else float(floor_guess)
    hi = y[y > floor + 2.0]
    if len(hi) < MIN_CEILING_POINTS:
        return floor, None, int(len(hi))
    h, e = np.histogram(hi, bins=np.arange(hi.min(), hi.max() + 0.01, 0.005))
    c = e[np.argmax(ndi.gaussian_filter1d(h.astype(float), 1))] + 0.0025
    near = hi[np.abs(hi - c) < 0.015]
    return floor, float(np.median(near)), int(len(near))

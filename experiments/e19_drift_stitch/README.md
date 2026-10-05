# e19 — drift ablation: raw ARKit placement vs plane-anchored stitch (GT-free)

```
uv run python experiments/e19_drift_stitch/stitch.py single_scan_with_ceiling V2      # v1
uv run python experiments/e19_drift_stitch/stitch_v2.py single_scan_with_ceiling V2   # v2 (ingredients separated)
```

Room shapes come from all data; each visit-split half (A = even, B = odd visits) places every room
where its own raw poses put it (ICP of the half's room cloud onto the full room cloud). Rooms with
one visit are excluded (no independent second placement). Metric: disagreement between the two
halves' layouts after one best rigid 2-D fit — centroid-based and corner-based (rotation-sensitive).

| Placement | centroid median / max (mm) | corners median / p90 / max (mm) |
|---|---|---|
| A raw ARKit | 46.8 / 88.9 | 54.8 / 96.8 / 115.8 |
| B1 common Manhattan yaw | 46.8 / 88.9 | 66.0 / 84.4 / 93.8 |
| B2 B1 + doorway jamb constraints | = B1 (0 doorway constraints measurable) | = B1 |
| B3 B2 + shared walls, one common thickness | 38.9 / 103.1 | 69.2 / 104.6 / 120.0 |
| B4 B2 + shared walls, per-wall thickness, Huber | 74.8 / 131.7 | 90.5 / 111.7 / 127.0 |

Findings: no translation stitch reduced disagreement; the common-thickness assumption moved room 4
by ~19 cm in both halves (v1). Doorway jamb gaps were measurable from both sides for 0/4 doorways
(small rooms: gap reaches the corner; open door leaves fill the gap). Yaw snapping trades the
median (+11 mm) for the tails (−19 % max). Shared-wall gap spread is not used as a metric: it is the
quantity B3 optimises (v1 reported 120 → 0.3 mm, which is circular).

Caveats: 5 rooms, one capture, halves share the trajectory between visits; no GT.

# e21 — photo-tier scale: two-family ensemble and disagreement-based intervals [PSEUDO-GT]

`uv run python experiments/e21_photo_ensemble/analyze.py` (CPU only; reads e16 runs.jsonl).
All learned quantities are leave-one-room-out. 6 rooms, 300 runs, K = 2..8. ~3 % pseudo-GT floor.

| estimator | median \|err\| | p90 | within ±8 % | catastrophic > 25 % |
|---|---|---|---|---|
| MoGe-2 (FOV given) | 7.5 % | 19.1 % | 54 % | 7 |
| MapAnything, per-view EXIF focal correction | 6.2 % | 31.5 % | 61 % | 49 |
| MapAnything raw × LORO constant (≈ 1.45) | 5.2 % | 23.7 % | 67 % | 27 |
| **ensemble = √(MoGe × MA-const)** | **5.3 %** | **14.0 %** | 64 % | **0** |

Per-room ensemble within ±8 %: rooms 1/3/4/8 → 92/94/88/90 %; room 6 → 16 % (both models
low: −16 % / −6 %); room 7 (mirror bathroom) → 10 % (MA +25 %, MoGe −2 %).
Per-room bias correlation MoGe vs MA-const: r = 0.46 (DAv2 vs MoGe was 0.75).
Intervals (LORO): fixed ±14.3 % → 85 % coverage; disagreement-adaptive → 82 % at ±12.9 %;
per-room coverage in rooms 6/7 only 32–66 %: model disagreement does not flag the failing rooms.

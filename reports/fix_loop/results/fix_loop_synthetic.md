> **SYNTHETIC.** Exact geometry of a rendered capture: a pipeline smoke test, not a benchmark.

# Evaluation: fix_loop_synthetic (gt_kind=synthetic)

| capture | tier | gate results |
|---|---|---|
| before_fix | lidar | rooms_matched=2/2, walls_matched=8, walls_unmatched=0, rooms_topology_mismatch=0, rooms_matched_by_size_only=2 (identity unverified: errors may be optimistic), wall_rel_err_median=0.084, area_rel_err_median=0.251, ceiling_abs_err_median_m=0.001, ceiling_pass=True, ceiling_max_abs_err_m=0.002, opening_pass=False, interval_coverage=0.538, n_intervals=13 |
| after_fix | lidar | rooms_matched=2/2, walls_matched=8, walls_unmatched=0, rooms_topology_mismatch=0, rooms_matched_by_size_only=2 (identity unverified: errors may be optimistic), wall_rel_err_median=0.001, area_rel_err_median=0.001, ceiling_abs_err_median_m=0.001, ceiling_pass=True, ceiling_max_abs_err_m=0.002, opening_pass=False, interval_coverage=1.000, n_intervals=13 |

## Question fix_loop

| capture | variant | rooms matched | phantom rooms | walls (measured, in interval) | median abs wall err | scale ratio | shape err | doorway err | diagnostics |
|---|---|---|---|---|---|---|---|---|---|
| before_fix | before | 2/2 | 0 | 8, 4 | 252 mm | A 0.917, B 0.832 | A 9.1 %, B 20.0 % | +10 mm | upper walls True |
| after_fix | after | 2/2 | 0 | 8, 8 | 2 mm | A 0.999, B 0.999 | A 0.0 %, B 0.0 % | +10 mm | upper walls True |

Figure: `fix_loop_synthetic_walls.png` (predicted vs reference wall lengths with 90 % intervals); plans: `<run>/<capture>/plan.png`.

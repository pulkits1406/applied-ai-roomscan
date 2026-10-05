> **PSEUDO-GT.** Reference values come from our own LiDAR reconstruction, not a laser/tape.
> These numbers measure consistency between tiers, not accuracy. They are not the benchmark.

# Evaluation: pseudo_gt_lidar_v11 (gt_kind=pseudo_lidar)

| capture | tier | gate results |
|---|---|---|
| lidar_floor_only | lidar | rooms_matched=1/6, walls_matched=4, walls_unmatched=0, rooms_topology_mismatch=0, rooms_matched_by_size_only=1 (identity unverified: errors may be optimistic), wall_rel_err_median=0.179, area_rel_err_median=0.353, opening_pass=False, interval_coverage=0.000, n_intervals=5 |
| lidar_single_room | lidar | rooms_matched=1/6, walls_matched=0, walls_unmatched=6, rooms_topology_mismatch=1, rooms_matched_by_size_only=1 (identity unverified: errors may be optimistic), area_rel_err_median=0.229, opening_pass=False, interval_coverage=0.000, n_intervals=1 |
| photo_diverse_k6 | photo | rooms_matched=2/6, walls_matched=4, walls_unmatched=6, rooms_topology_mismatch=1, wall_rel_err_median=6.800, area_rel_err_median=29.219, ceiling_abs_err_median_m=0.427, walls_within_gate_frac=0.000, walls_max_rel_err=7.999, stitch_pass=False, stitch_fail_reason=3 room(s) not placed, interval_coverage=0.286, n_intervals=7 |
| photo_sweep_k8 | photo | rooms_matched=3/6, walls_matched=8, walls_unmatched=6, rooms_topology_mismatch=1, wall_rel_err_median=2.715, area_rel_err_median=0.443, ceiling_abs_err_median_m=0.157, walls_within_gate_frac=0.000, walls_max_rel_err=8.071, stitch_pass=False, stitch_fail_reason=4 room(s) not placed, interval_coverage=0.167, n_intervals=12 |
| video_with_ceiling | video | rooms_matched=2/6, walls_matched=4, walls_unmatched=6, rooms_topology_mismatch=1, rooms_matched_by_size_only=2 (identity unverified: errors may be optimistic), wall_rel_err_median=0.176, area_rel_err_median=0.129, walls_within_gate_frac=0.000, walls_max_rel_err=0.239, interval_coverage=0.667, n_intervals=6 |

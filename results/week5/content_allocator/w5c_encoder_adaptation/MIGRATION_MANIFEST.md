# Migration Manifest: Week 6 to Week 5-C (Encoder Adaptation)

**Date:** 2026-10-11
**Status:** VERIFIED (All SHA256 Checksums Identical)

This manifest documents the byte-level preservation of experimental results migrated from `results/week6/encoder_adaptation/` to `results/week5/content_allocator/w5c_encoder_adaptation/`.

| Old Path | New Path | Action | Size (Bytes) | SHA256 (Truncated) | Checksum Match |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `results/week6/encoder_adaptation/models/C1_seed_42.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C1_seed_42.pt` | Moved | 102,175,834 | `f9c00ddb611ed779...` | PASS |
| `results/week6/encoder_adaptation/models/C1_seed_123.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C1_seed_123.pt` | Moved | 102,175,977 | `9cc25442cd61c045...` | PASS |
| `results/week6/encoder_adaptation/models/C1_seed_2026.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C1_seed_2026.pt` | Moved | 102,176,120 | `add760670b49b741...` | PASS |
| `results/week6/encoder_adaptation/models/C2_seed_42.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C2_seed_42.pt` | Moved | 102,175,834 | `b578596e72e1b0bb...` | PASS |
| `results/week6/encoder_adaptation/models/C2_seed_123.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C2_seed_123.pt` | Moved | 102,175,977 | `462881678035f320...` | PASS |
| `results/week6/encoder_adaptation/models/C2_seed_2026.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/models/C2_seed_2026.pt` | Moved | 102,176,120 | `c0ff6723e99fd743...` | PASS |
| `results/week6/encoder_adaptation/predictions/dev_predictions.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/predictions/dev_predictions.csv` | Moved | 6,282 | `466309e389a8f8cd...` | PASS |
| `results/week6/encoder_adaptation/tables/w6_main_results.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_main_results.csv` | Renamed | 792 | `52421007f130f12c...` | PASS |
| `results/week6/encoder_adaptation/tables/w6_seed_results.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_seed_results.csv` | Renamed | 919 | `1dacf7a24f68543c...` | PASS |
| `results/week6/encoder_adaptation/tables/w6_multilabel_metrics.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_multilabel_metrics.csv` | Renamed | 537 | `9e6e769f3f11e855...` | PASS |
| `results/week6/encoder_adaptation/tables/w6_tau_selection.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_tau_selection.csv` | Renamed | 386 | `c46177bb1e7fa185...` | PASS |
| `results/week6/encoder_adaptation/tables/w6_capacity_comparison.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/tables/w5c_capacity_comparison.csv` | Renamed | 283 | `051de0476a76de19...` | PASS |
| `results/week6/encoder_adaptation/bootstrap/w6_bootstrap_comparisons.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/bootstrap/w5c_bootstrap_comparisons.csv` | Renamed | 757 | `3c0d6eefe0f82abf...` | PASS |
| `results/week6/encoder_adaptation/figures/w6_pareto.png` | `results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_pareto.png` | Renamed | 191,900 | `bb25b1c5f2d3e982...` | PASS |
| `results/week6/encoder_adaptation/figures/w6_selected_k_distribution.png` | `results/week5/content_allocator/w5c_encoder_adaptation/figures/w5c_selected_k_distribution.png` | Renamed | 84,055 | `3ffa0f3124266a2c...` | PASS |
| `results/week6/encoder_adaptation/diagnostics/representation_shift.csv` | `results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/representation_shift.csv` | Moved | 738 | `cf1d80af3677bfb5...` | PASS |
| `results/week6/encoder_adaptation/diagnostics/representation_adaptation_health.json` | `results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/representation_adaptation_health.json` | Moved | 2,949 | `ee0526818b4dd074...` | PASS |
| `results/week6/encoder_adaptation/diagnostics/smoke/smoke_test_results.json` | `results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/smoke/smoke_test_results.json` | Moved | 1,646 | `009e656c723dcf33...` | PASS |
| `results/week6/encoder_adaptation/diagnostics/smoke/smoke_checkpoint.pt` | `results/week5/content_allocator/w5c_encoder_adaptation/diagnostics/smoke/smoke_checkpoint.pt` | Moved | 102,176,692 | `5bca9cc89d256d8c...` | PASS |


## Verification Summary
- Total Files Migrated: 19
- Checksums Verified 1:1: True
- Experimental Numerical Outputs: Byte-for-byte identical.

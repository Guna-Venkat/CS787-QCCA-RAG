# BEA_V2 final allocation-only resolution — terminated

The amendment was saved before the first new allocation in
`results/post_w12/bea_v2/v3/calibration/resolution/AMENDMENT.md` and its hashed JSON companion.

> The initial preregistered grid identified the qualifying boundary between 5e-5 and 1e-4, but no grid point satisfied all mechanical constraints. A single prespecified resolution sweep of 6e-5,7e-5,8e-5,9e-5 will be performed. The original eligibility rules remain unchanged. Select the smallest threshold satisfying every original criterion. If none qualifies, terminate BEA_V2 calibration.

Exactly the original nine queries (three per dataset), both budgets, and four new thresholds were evaluated: **72 allocation-only conditions**. The zero-threshold coverage reference was loaded from the original saved allocations. No reader, Qwen generation, F1, gold answers/evidence or held-out LOO labels were inspected. No predictor was refitted.

| Threshold | Threshold stops | All dataset/budget cells pass? | Global mean utilization | Eligible? |
|---|---:|---|---:|---|
| 0.00006 | 3/18 = 16.67% | Yes | 85.55% | No: stop rate below 20% |
| 0.00007 | 3/18 = 16.67% | Yes | 84.56% | No: stop rate below 20% |
| 0.00008 | 3/18 = 16.67% | Yes | 84.38% | No: stop rate below 20% |
| 0.00009 | 5/18 = 27.78% | No | 83.03% | No: SciFact @1024 utilization below 20% |

At 0.00009, SciFact @1024 averages **15.30% budget utilization**, below the unchanged 20% floor. Its median selected unit count is 3; trivial-context fraction is 1/3; coverage retention is 99.83%. The failed condition is mean utilization. All thresholds produce zero empty contexts. Global coverage retention is 100% for the first three thresholds and 99.97% for 0.00009. Near-full allocation fraction is 72.22% for the first three and 66.67% for 0.00009, below the 80% ceiling.

**Decision: CALIBRATION_FAILED_TERMINATED. No threshold is frozen and no 300-answer reader pilot was launched.** No further calibration or relaxation is permitted under this amendment. On this small fixed sample, the final grid cannot simultaneously achieve sufficient threshold stopping and healthy evidence exposure across every dataset/budget cell. This is an allocation-calibration result, not an answer-quality result or proof that every possible threshold would fail.

The original failed calibration, nine-query membership, all predictor files, folds, hyperparameters, aspects, P_help scoring, lambda/eta, and eligibility rules remain unchanged. Per-condition allocations, immutable traces, all 24 dataset/budget checks, the amendment and the final result are saved under `results/post_w12/bea_v2/v3/calibration/resolution/`. The terminal receipt is `results/post_w12/bea_v2/v3/CALIBRATION_TERMINATED.json`.

The notebook now defaults `RUN_PILOT=False` and displays the terminal report before stopping, ahead of any attribution audit, training, calibration or reader execution. Research and historical artifacts were not modified.

Verification: **27 tests passed**, including exact nine-query/four-threshold coverage, original-file checksums, unchanged eligibility evaluation, termination/no-reader checks, and notebook safety. All notebook code cells compile. The artifact-protection audit verified **771 protected files unchanged**. Two existing scikit-learn deprecation warnings are unrelated to allocation.

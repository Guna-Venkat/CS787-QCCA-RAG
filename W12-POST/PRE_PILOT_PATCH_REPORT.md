# BEA-QBGF pre-pilot patch report

**Ready for the manually launched pilot.** Tests, nine-query allocation-only calibration and the new BEA freeze are complete. **No answer-generation pilot was started.**

## Corrections

- FULL utility now recomputes redundancy and coverage sequentially after each accepted hypothetical sentence; it no longer independently sums optimistic gains. FULL still pays the exact entire-parent replacement cost and competes globally with sentence actions.
- Two positive parent-sentence selections remain the eligibility **design assumption**, not a theorem. Traces and final analysis record FULL eligibility, consideration, selection, cost/low-gain/threshold rejection, lost competition and one-sentence blocking.
- Notebook/CLI enforce **tests → allocation calibration → final freeze → pilot**. Missing compatible work resumes automatically. Failed tests or calibration block generation. No answer F1 or gold evidence selects the threshold.
- New outputs use `results/post_w12/bea_qbgf/v3`; v1 delivery metadata and v2 failed-test records remain intact. Derived caches are ignored by Git. Frozen Week12 definitions, splits, controls and results remain unchanged.

## Tests and allocation smoke

**182 passed, 3 explicitly deselected**, including all **15 BEA tests**. The three historical notebook-default assertions conflict with completed full-run notebook settings; their exclusion was approved by the user. Original test/notebook files were not edited. Exact exclusions and the successful command are in `calibration/TESTS.json`; the earlier unfiltered run reported 182 passed / 3 failed.

Smoke: three exact pilot VAL IDs per dataset, budgets 256/1024; **18 allocations per threshold, 108 total**. No reader generation or answer scoring. Existing native VAL adapters parse records, but evaluation payloads are removed before allocation. The smoke took about **14.5 minutes**, mainly rebuilding missing full-corpus SciFact embeddings; those caches are retained for the pilot.

|   Threshold |   Threshold stops (%) |   Budget stops (%) |   Mean budget use (%) |   Cosine-coverage retention (%) | Eligible   |
|------------:|----------------------:|-------------------:|----------------------:|--------------------------------:|:-----------|
|      0.001  |                  0    |             100    |                 98.34 |                          100    | False      |
|      0.0025 |                  5.56 |              94.44 |                 98.31 |                          100    | False      |
|      0.005  |                 16.67 |              83.33 |                 95.91 |                          100    | False      |
|      0.01   |                 72.22 |              27.78 |                 67.12 |                           99.61 | True       |
|      0.02   |                100    |               0    |                 13.94 |                           96.07 | True       |
|      0.04   |                100    |               0    |                  2.98 |                           82.47 | False      |

**Frozen choice: `tau_stop=0.01`, `tau_expand=0.01`**; `tau_context=0.05`, `lambda_red=eta_cov=0.5`, minimum positive parent selections=2. This is the smallest threshold passing the preregistered allocation rule. At `.001`, all allocations stopped for budget fit; at `.01`, 72.2% stopped for low marginal value, mean utilization was 67.1%, and mean cosine-coverage retention was 99.6%. These are surrogate allocation diagnostics, **not answer-quality improvements**. Nine queries provide limited calibration evidence.

## FULL diagnostics at the frozen threshold

Each dataset/budget cell has three queries. All 18 allocations reached FULL eligibility; only **2/18 (11.1%)** actually chose FULL, both HotpotQA. Counts below are candidate-step events, so the same parent can be considered repeatedly. Low gain and low gain-per-token are logged separately.

| Dataset   |   Budget |   Eligible queries |   Considered |   Chosen |   Low-gain rejection |   Cost rejection |   Threshold rejection |
|:----------|---------:|-------------------:|-------------:|---------:|---------------------:|-----------------:|----------------------:|
| HotpotQA  |      256 |                  3 |          111 |        1 |                    0 |               71 |                    25 |
| HotpotQA  |     1024 |                  3 |          161 |        1 |                    0 |                0 |                   145 |
| QASPER    |      256 |                  3 |          132 |        0 |                    0 |              131 |                     1 |
| QASPER    |     1024 |                  3 |          238 |        0 |                    0 |                0 |                   238 |
| SciFact   |      256 |                  3 |           89 |        0 |                    0 |               89 |                     0 |
| SciFact   |     1024 |                  3 |          481 |        0 |                    0 |               75 |                   406 |

Activation alone does not establish benefit. FULL was blocked mainly by exact cost at 256 and gain-per-token threshold at 1024; the complete pilot must establish downstream quality.

## Freeze and next action

- Verified **771 protected files unchanged**; original final freeze remains byte-identical with SHA256 `d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b`.
- Calibrated BEA identity: `7b6623f9bbc2ba1e0800d21485ad61b373f6d1f1e49beb401b53e8ae53185d7a`; stored in `results/post_w12/bea_qbgf/v3/BEA_QBGF_V1_FREEZE.json`. This exploratory freeze does not replace the historical final-method freeze.
- Plan: **900 logical pilot rows**, 0 completed local answer rows; upper bound 900 new answers before compatible legacy/exact-prompt reuse.
- Open **`W12-POST/BEA_QBGF.ipynb`**, select **`sara_env`**, and use **Run All**. Leave `RUN_PILOT=True`, `RUN_TESTS=True`, `AUTO_RUN_MISSING=True`, `RESUME=True`; leave overwrite/regeneration, expansion and attribution disabled. Completed smoke inputs and embeddings are reused; missing pilot work runs.
- Allocation-only artifacts: `calibration/ALLOCATION_ONLY_REPORT.md`, `threshold_summary.csv`, `step_marginal_distribution.csv`, `full_diagnostics.csv`, per-allocation traces and test receipt.

Branch: **gagan**. No confirmation or QASPER TEST outcomes accessed. No full pilot, expanded VAL or attribution run started.

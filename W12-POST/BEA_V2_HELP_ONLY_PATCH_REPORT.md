# BEA_V2 help-only pre-pilot patch

**Both requested changes are implemented. Calibration is complete, but the 300-answer pilot remains blocked: none of the ten tested global thresholds satisfies every preregistered constraint. No final ready-to-run threshold or BEA_V2 freeze was fabricated. No reader answers were generated.**

## Changes

1. Primary value is now exactly `U = P_help × max(0, Rel + 0.5 ΔCoverage − 0.5 Redundancy)`. P_harm and P_help−P_harm remain diagnostic fields only; they affect neither allocation nor the primary predictor-quality gate. Tests explicitly verify that changing P_harm cannot alter evidence IDs, transition values or stopping. Source states, sequential IG recomputation, exact token costs and FULL eligibility are unchanged.
2. Calibration uses **nine unique development queries: three/dataset**, both budgets and one common threshold. All allocation checks use cached semantic features and held-out predictions; no Qwen, answer F1, gold evidence or held-out LOO labels enter calibration. Hyperparameters, features and cross-fit grouping were not changed.

All **15 fold predictor bundles and the fold membership file are byte-identical** to the previous version. Previous freezes/models/calibrations remain untouched. Active outputs are `results/post_w12/bea_v2/v3/`; the original help-minus-harm source/config/notebook snapshot is retained under `W12-POST/archive/bea_v2_help_minus_harm_prepatch/`.

## Rule registered before allocation

- Grid: 0, .00001, .000025, .00005, .0001, .00025, .0005, .001, .0025, .005. Lower values were included before this run to cover the probability-weighted numerical scale.
- Globally: **zero empty contexts**, at least **20% threshold stops**, at most **80% near-full allocations** (≥95% utilization), and ≥85% mean cosine-coverage retention against threshold zero.
- **Every dataset/budget cell:** mean utilization ≥20%; median selected evidence units ≥2; no more than one-third trivial contexts (<10% utilization or <2 units); ≥90% mean cosine-coverage retention.
- Select the smallest qualifying global threshold. If none qualifies, **NEEDS_REVIEW**. These checks do not impose minimum tokens on the allocator, and they are heuristics rather than evidence that a short context is wrong.

The sample, complete rule and source/model identities were saved before allocation in `calibration/PREREGISTRATION.json`.

## Allocation-only results

Nine queries × two budgets × ten thresholds = **180 cached allocations**. No new reader predictions or quality measurements.

|   threshold |   n |   threshold_stop_fraction |   budget_stop_fraction |   mean_utilization |   near_full_fraction |   empty_fraction |   coverage_retention | all_cells_eligible   |   failed_cells | eligible   |
|------------:|----:|--------------------------:|-----------------------:|-------------------:|---------------------:|-----------------:|---------------------:|:---------------------|---------------:|:-----------|
|    0.000000 |  18 |                  0.000000 |               1.000000 |           0.972493 |             0.888889 |         0.000000 |             1.000000 | True                 |              0 | False      |
|    0.000010 |  18 |                  0.111111 |               0.888889 |           0.941515 |             0.777778 |         0.000000 |             1.000000 | True                 |              0 | False      |
|    0.000025 |  18 |                  0.111111 |               0.888889 |           0.906087 |             0.777778 |         0.000000 |             1.000000 | True                 |              0 | False      |
|    0.000050 |  18 |                  0.166667 |               0.833333 |           0.867459 |             0.722222 |         0.000000 |             1.000000 | True                 |              0 | False      |
|    0.000100 |  18 |                  0.333333 |               0.666667 |           0.819933 |             0.555556 |         0.000000 |             0.999715 | False                |              1 | False      |
|    0.000250 |  18 |                  0.500000 |               0.500000 |           0.685547 |             0.388889 |         0.000000 |             0.997768 | False                |              1 | False      |
|    0.000500 |  18 |                  0.611111 |               0.388889 |           0.548394 |             0.333333 |         0.055556 |             0.942213 | False                |              1 | False      |
|    0.001000 |  18 |                  0.833333 |               0.166667 |           0.339952 |             0.166667 |         0.111111 |             0.885339 | False                |              3 | False      |
|    0.002500 |  18 |                  1.000000 |               0.000000 |           0.062066 |             0.000000 |         0.222222 |             0.726561 | False                |              6 | False      |
|    0.005000 |  18 |                  1.000000 |               0.000000 |           0.039280 |             0.000000 |         0.333333 |             0.620431 | False                |              6 | False      |

The relevant tradeoff is narrow but unresolved on the tested grid:

- **0.00005:** all six cells pass their exposure/coverage checks; no empty context; 100% cosine-coverage retention. Threshold stopping occurs for 3/18 allocations (**16.7%**), below the preregistered 20% requirement.
- **0.0001:** threshold stopping is 6/18 (**33.3%**) and no context is empty, but SciFact@1024 mean utilization is **15.3%**, below its 20% cell-level guard. This is a mechanical guard failure, not measured evidence of answer harm.
- **0.00025:** the previous numeric threshold now gives SciFact@1024 mean utilization **4.4%**, median one unit and three trivial contexts; it is rejected.
- Lower/higher tested values do not satisfy the full rule. This does **not** prove that no intermediate threshold could work; such values were not tested. No grid refinement or cutoff relaxation was performed after inspecting these results.

### Cell-level detail around the boundary

|   threshold | dataset   |   budget |   mean_utilization |   median_units |   trivial_fraction |   coverage_retention |   threshold_stop_fraction | eligible   | failure_reasons      |
|------------:|:----------|---------:|-------------------:|---------------:|-------------------:|---------------------:|--------------------------:|:-----------|:---------------------|
|    0.000050 | HotpotQA  |      256 |           0.960938 |       8.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000050 | HotpotQA  |     1024 |           0.976237 |      11.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000050 | QASPER    |      256 |           0.975260 |      10.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000050 | QASPER    |     1024 |           0.980143 |      22.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000050 | SciFact   |      256 |           0.962240 |       6.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000050 | SciFact   |     1024 |           0.349935 |       8.000000 |           0.000000 |             1.000000 |                  1.000000 | True       |                      |
|    0.000100 | HotpotQA  |      256 |           0.960938 |       8.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000100 | HotpotQA  |     1024 |           0.945638 |      11.000000 |           0.000000 |             1.000000 |                  0.333333 | True       |                      |
|    0.000100 | QASPER    |      256 |           0.975260 |      10.000000 |           0.000000 |             1.000000 |                  0.000000 | True       |                      |
|    0.000100 | QASPER    |     1024 |           0.964193 |      21.000000 |           0.000000 |             1.000000 |                  0.333333 | True       |                      |
|    0.000100 | SciFact   |      256 |           0.920573 |       6.000000 |           0.000000 |             1.000000 |                  0.333333 | True       |                      |
|    0.000100 | SciFact   |     1024 |           0.152995 |       3.000000 |           0.333333 |             0.998290 |                  1.000000 | False      | low_mean_utilization |

## Tests, integrity and execution

**22 CPU tests passed.** These include the existing allocator/cross-fit/reporting checks plus P_harm-invariance, zero-help behavior, global-versus-cell collapse detection, zero-empty enforcement and cell-level coverage checks independent of answer/gold columns. The existing nonfatal sklearn penalty deprecation warning remains.

Verified **123 pre-existing V2/freeze/model/calibration artifacts byte-identical**, plus the original 771-file protected research manifest. No V1 process was interrupted or GPU model loaded. No full pilot allocation or reader answer row exists in the new namespace. The source-weight change is motivated by previously inspected predictor diagnostics; this remains a developmental experiment, not fresh confirmation.

The notebook has been updated to the help-only formulation and nine-query protocol. It deliberately stops at calibration/freeze if run now. **Do not launch the 300-answer stage yet.** `PILOT_READINESS.json` records the blocked protocol and model hashes; `BEA_V2_FREEZE.json` is absent because no threshold qualifies. Existing older freezes are not reused for the new score.

No further method changes, threshold-rule relaxation, fine-grid search, pilot, new attribution or unseen expansion was performed. A decision about the failed calibration is needed before a valid final inference freeze can be created.

Branch: **gagan**.

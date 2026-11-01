# W9-A FIT freeze audit

BRANCH = gagan. CPU-only saved-artifact audit; no model generations or confirmation-split selection.

Complete grid: 298,566 rows, 1,746 queries, 656 papers. Recall averages over 1,682 queries with annotated evidence; missing evidence is not a successful retrieval.

## Decision

Frozen **FIXED-5/SEMANTIC@20** using the existing preregistered global .003 recall tolerance, then retrieved-token cost. No post-hoc SEM-4 preference was substituted. The minimum eligible token cost is unique, so mixed historical/optimized timing measurements do not affect this decision.

Global best recall: 0.878716. Selected recall: 0.878716; retrieved tokens: 5600.5; coarse density: 0.0504.

## Chunk size and saturation

| Representation | Pooled mean / median chunk tokens | Mean / median chunks per paper | Queries with whole pool at W20 |
|---|---:|---:|---:|
| ORIG | 189.9 / 222.0 | 29.13 / 27 | 26.5% |
| FIXED-4 | 201.3 / 205.0 | 27.49 / 26 | 33.3% |
| SEM-4 | 315.5 / 205.0 | 18.08 / 15 | 69.8% |
| SEM-5 | 715.6 / 1022.5 | 8.49 / 7 | 97.1% |
| SEM-7 | 725.2 / 1024.0 | 8.37 / 7 | 97.0% |
| SEM-8 | 909.0 / 1024.0 | 6.84 / 6 | 98.3% |
| FIXED-5 | 934.6 / 1022.0 | 5.92 / 6 | 99.4% |

Statistics use one representative cached partition per paper/representation (the same paper deduplication principle used to match fixed sizes). All ORIG query partitions were checked for equal length/count statistics. SEM/FIXED boundaries are question-independent by construction. Query-weighted availability uses FIT paper query counts and agrees with the selected actual retrieval rows.

SEM-5/7/8 are large, few-chunk regimes: around 97–98% of queries exhaust the entire pool at W20. W40 adds little because few extra chunks remain, rather than proving better candidate discrimination. Retrieved-token totals include semantic overlap/repetition.

## The provisional SEM-4/RRF@20 point

- Recall 0.861474, tokens 5285.2, density 0.0697; it is on the three-objective Pareto frontier.
- Versus ORIG/RRF@20: +6.956 recall percentage points, but 44.0% more retrieved tokens.
- W20→W40 adds 1.249 recall points and 420.2 tokens (8.0%). Mean actual candidates rise 14.89→17.51, not by 20. Already 69.8% of queries exhaust the pool at W20.

## Length matching and semantic-boundary interpretation

The existing FIXED-i control matches the pooled median SEM-i chunk length, not the complete length distribution. SEM-4 has the same 205-token median as FIXED-4 but a much longer tail: pooled means 315.5 vs 201.3 tokens. A same-width gain therefore does not isolate semantic boundaries from length/budget.
At approximately matched retrieved-token budgets: SEM-4/RRF@20 uses 5285.2 tokens and recall 0.861474; FIXED-4/RRF@40 uses 5209.5 and 0.856124; ORIG/RRF@40 uses 5212.4 and 0.858502. The SEM-4 recall advantage is only 0.535 and 0.297 points respectively, with lower overlap density. These are near-budget comparisons, not exact causal length-matched experiments or significance claims.

All nominal median-matched SEM/FIXED pairs are saved in `length_matched_fixed_vs_semantic.csv`. All 171 configurations and the 3-D frontier are saved separately; plots encode density in color. Density is the project’s paragraph/word-overlap proxy, not exact evidence-token annotation or answer faithfulness.

## W9-B readiness and interpretation

W9-B already holds DraftSemantic fixed and evaluates K_E ∈ {0,1,2,4}. K=0 reuses the canonical initial draft. K>0 uses raw selected chunks at this stage; compact extraction is tested later in D/E. `surface_analysis()` implements the smallest exact-maximum K and the smallest K within .01 F1 of the maximum. No predictor is trained by B.
The frozen FIXED-5 chunks have median 1,022 tokens; a “unit” here is not a 205-token or extracted evidence item. 256/1746 queries (14.66%) have fewer than four candidates. Report actual counts and the full-four-candidate subgroup before interpreting nominal K=4 or concluding one compact unit suffices. The B notebook includes that analysis.
A high K=2/4 epsilon-oracle mass would motivate investigating G, but does not replace G’s separate preregistered headroom/cost gate. B oracle results are retrospective ceilings, not deployable predictors.

The canonical frozen A artifact passes `load_frozen()` and B dependency/protocol checks. No B generation run was launched. Run manually:

```bash
./sara_env/bin/python scripts/week9/run_w9b_ke_surface.py --split fit --config configs/week9/w9b.yaml --resume
```

Do not change A after B starts. A different compactness-constrained choice would be a new FIT-only amended experiment/output root, rather than an overwrite of this frozen decision.

## Notebook verification

W9-B executed successfully with RUN_FULL=False and RUN_SMOKE=False. Its new oracle/available-pool analysis was also checked on a temporary fixture outside results/. The existing epsilon-oracle regression test passes. No runtime source files were changed by this audit, and frozen A still validates against current sources and artifacts.

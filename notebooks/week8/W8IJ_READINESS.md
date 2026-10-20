# W8-I / W8-J execution and scope

Use the `sara_env` notebook kernel. W8-H and earlier completed experiment sources/results remain unchanged.

## W8-I — `week8i_ablations.ipynb`

Run all cells. `RUN_FULL=True` is the default. This is CPU reporting and counterfactual replay; it does not regenerate answers or retrain models. The 342-query report contains the k4 baseline, eight controller/selection replays, and all eight W8-H generator/revision combinations. Exact prompt costs and frozen G0 answers must reproduce the measured H run. Each ablation is compared with the frozen controller using paired paper-clustered intervals; all systems are also compared with k4.

Outputs: `results/week8/w8i_ablations/full/benchmark.csv`, `paired_paper_intervals.csv`, `query_results.jsonl`, `quality_cost.png`, `protocol.json`, `summary.json`.

## W8-J — `week8j_final_dev_evaluation.ipynb`

The default `RUN_FULL=False` runs preflight and the real 20-query GPU smoke. The sample uses only question IDs and seed 42; it must exercise both STOP and PROMOTE. After that succeeds, set `RUN_FULL=True` in the first cell and run all cells to evaluate all 231 DEV queries. The verified smoke cache is reused. At least 18 GiB free GPU memory is required when a model needs loading. Do not launch competing GPU experiments.

All three W8-G adapters and original SARA are retained independently, each in G0 and G1. The controller always observes a fresh original-SARA draft. Static semantic k2/k4/k5/k6/k8 answers come from the real historical DEV response surface and are rescored using the current scorer. Serialized prompt costs are rebuilt with the checkpoint tokenizer. New k4 drafts must reproduce the saved canonical text. Duplicate or near-tied passages retain the recorded k4 allocation and must pass the established MiniLM tolerance check.

Outputs: `results/week8/w8j_final_dev/{smoke,full}/`, including generated predictions, live decisions, source provenance, benchmark, paired intervals, and quality/cost plot. Source changes or modified artifacts reject stale caches. Smoke and full results are separate. The initial five-query STOP-only smoke is archived and does not qualify the promotion branch.

## Scientific limits

- INTERNAL-VAL and DEV were inspected previously. These are development evaluations, not untouched final-test evidence. TEST is unused.
- No DEV tuning, retraining, checkpoint selection, or automatic winner selection occurs.
- DAFiR-2 has no frozen second-state controller; QCLA/controller stacking changes the gate's draft distribution. Both remain explicitly unsupported rather than receiving placeholder scores.
- Removing a HELP/HURT safety filter does not establish the effect of retraining without a corresponding loss. Replays are labelled separately from new inference.
- Cumulative prompt tokens measure prompt-processing volume, not FLOPs or deployed end-to-end latency. Timings exclude cached retrieval/SFR and cold model setup.
- Intervals use 2,000 paired paper-cluster bootstrap resamples with query-weighted means. Multiple comparisons are exploratory, with no confirmatory significance claims.
- Old hard-coded I/J smoke tables were removed from active tables; their identical archived copies and retirement ledger are preserved.

## Verification completed on 2026-10-20

- All 107 Week 8 tests passed.
- W8-I notebook executed all cells, including the full 342-query report.
- W8-J preflight verified all 231 DEV records and semantic allocations. Its real 20-query GPU smoke passed all eight generator/revision combinations with 14 STOP and 6 PROMOTE queries.
- W8-J notebook executed all cells with `RUN_FULL=False` and verified its saved measured smoke artifacts. The full DEV run has not been executed.
- Artifact checksums and notebook structure passed verification; completed W8-H sources/results remain unchanged.

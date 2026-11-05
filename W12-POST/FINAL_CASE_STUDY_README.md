# Final query-level qualitative analysis notebook

Open **`W12-POST/FINAL_QUERY_LEVEL_CASE_STUDY.ipynb`** with `sara_env` and choose **Run All**. This is a CPU-only saved-result analysis. The notebook defaults `RUN_ANALYSIS=True` and has no experiment/generation stage. No setting enables allocation, training, new embeddings or reader inference.

Prerequisite: finish the existing post-hoc BEA_V2 study first. All 1,200 saved results must exist for the original 50 development queries per dataset × two budgets × four methods. Missing results produce a clear error; this notebook will never regenerate them. Source code and original protocol artifacts remain unchanged.

The notebook:

1. verifies the saved post-hoc identity and source/result checksums;
2. builds a 300-row aligned query/budget table with all four methods;
3. selects 18 deterministic primary case slots and up to three substantive budget-reversal cases, disclosing duplicates/nonqualifying fallbacks;
4. reconstructs saved contexts, source membership, fidelity choices, selected-transition IG/P_help/marginal terms and stopping traces where available;
5. computes retrospective support retention from native gold annotation offsets, keeping unmapped annotations and absent SciFact NEI support explicit;
6. creates four-column case dossiers, automatic observed/candidate patterns, a persistent human-review table, full-cohort descriptive diagnostics and a conservative final interpretation.

No answer/gold data was used by the allocator. Gold is read only when **you execute** this retrospective analysis. QASPER TEST and confirmation cohorts are excluded by the existing native cohort loader.

## Output

Under `results/post_w12/final_case_study/`:

- `CASE_SELECTION.csv` and `CASE_SELECTION_RULES.json`
- `CASE_STUDIES.md`
- `FAILURE_MODE_SUMMARY.csv`
- `GOLD_EVIDENCE_RETENTION.csv` and `GOLD_EVIDENCE_RETENTION_SUMMARY.csv`
- `BUDGET_REVERSALS.csv`
- `FINAL_INTERPRETATION.md`
- `ALIGNED_PER_QUERY.csv`, `HYPOTHESIS_DIAGNOSTICS.csv`
- `AUTOMATIC_PATTERN_CANDIDATES.csv`, `MANUAL_REVIEW.csv`
- source/analysis identity and completion receipts

## Human review

Automatic fragmentation/pruning/reader-failure flags are **candidate explanations**, not established causes. The initial interpretation says explicitly when no human-confirmed mechanisms exist.

After reading dossiers, edit `MANUAL_REVIEW.csv`: confirm/reject existing candidates or add supported labels from the documented label list. `CONFIRMED` requires a reviewer, rationale and exact source-unit/trace references. Do not edit query/case membership. Then rerun the notebook: derived reports refresh, the manual-review file is preserved, and no experiment is repeated.

Each frequency table reports case-slot and unique query/budget counts. These examples were selected using outcomes; the counts are not population estimates. Full-cohort diagnostic means refer only to the original 50-query-per-dataset development cohort.

Gold evidence recall uses fully retained non-whitespace content of mapped annotations; parent recall refers to annotated native documents; gold token coverage uses fully covered non-whitespace tokenizer-offset tokens. Whitespace-only gaps between exact source sentences do not count as lost evidence. Alternative QASPER occurrences use the best source match. Pooled support annotations are not a minimal necessary evidence set. Unavailable mapping is not a retrieval failure.

No missing candidate probabilities, rejected semantic terms or traces are invented. Saved P_help scores are shown where present. Redundancy over-penalization and P_help miscalibration require manual evidence; aggregate correlation alone does not establish them. QASPER 256-fragmentation and 1024-stopping are treated as separate hypotheses. SciFact case scores are correctness; aggregate quality remains three-class macro-F1.

## Implementation verification

Only static syntax/file-scope inspection is performed during implementation. **No notebook, analysis, experiment or tests were executed by the agent.** Synthetic tests are supplied for optional manual execution:

```bash
CUDA_VISIBLE_DEVICES='' sara_env/bin/python -m pytest W12-POST/final_case_study_tests -q
```

The official BEA_V2 calibration remains **CALIBRATION_FAILED_TERMINATED**. This notebook neither changes that status nor promotes/tunes any method.

# W12-POST — BEA-QBGF

**POST_W12_EXPLORATORY: VAL only.** The historical final policy remains SFR top10 → FIXED_SENTENCE → Qwen3-8B BASE_FP16. Confirmation and QASPER TEST are excluded.

## Run

Open `BEA_QBGF.ipynb` with the `sara_env` kernel and use **Run All**. The notebook runs these stages in order:

1. Verify frozen dependencies and recover the exact 50-per-dataset K/L pilot cohort.
2. Run synthetic BEA tests and relevant unchanged Week12 tests.
3. Reuse/run nine-query allocation-only calibration, with no answer generation or F1 scoring.
4. Freeze the selected global parameters. Failed tests or `NEEDS_REVIEW` calibration block the pilot.
5. Prepare missing contexts with SFR; release SFR before loading Qwen.
6. Generate only missing compatible pilot rows; analyze the complete paired surface.

The pilot has **150 queries × two budgets × three methods = 900 logical outputs**. Exact prompt reuse may reduce physical answer calls. The implementation agent has not started these answer generations. Expansion and attribution remain separate opt-ins.

| Setting | Default | Meaning |
|---|---|---|
| RUN_PILOT | True | Manual Run All runs/reuses the pilot after all gates pass |
| RUN_TESTS | True | Validate before calibration; calibration enforces a compatible test receipt |
| AUTO_RUN_MISSING | True | Infer missing work from compatible caches |
| REUSE_COMPATIBLE_CACHE | True | Validate exact source/evidence/prompt/model identities |
| RESUME | True | Preserve completed compatible work |
| OVERWRITE_EXISTING / FORCE_REGENERATE | False | Incompatible results require a fresh output version |
| RUN_EXPANDED / RUN_ATTRIBUTION | False | Require PROMISING pilot and explicit manual opt-in |

Current outputs use `results/post_w12/bea_qbgf/v3/`. Earlier v1 delivery metadata and v2 failed validation records are retained. Historical Week12 files remain read-only. Derived extension caches are ignored by Git; compact reports, calibration summaries and freeze identities remain visible.

## Allocation and calibration

Sentence gain is query cosine + `0.5 × new aspect coverage − 0.5 × redundancy`. Candidates compete by gain per exact incremental serialized token. Scores are recomputed after every selection; presentation restores source order. Query aspects are deterministic query-only fragments, not verified facts.

FULL eligibility after two positive selections from the same parent is a **design assumption**. Its gain uses sequential hypothetical additions in source order, recomputing redundancy and coverage after every accepted sentence. The exact cost includes the entire parent replacement, including context not credited with gain. FULL still competes globally with sentence actions. No extraction LM is used.

Calibration uses three existing pilot VAL queries per dataset and both budgets. The preregistered threshold grid is `{.001, .0025, .005, .01, .02, .04}`. Choose the smallest global threshold with ≥20% threshold stops, ≤80% allocations using ≥95% budget, ≤10% empty evidence, and ≥85% mean aspect-coverage retention relative to `.001`. These are allocation heuristics, not quality guarantees. `tau_expand` is at least the selected stopping threshold. No eligible value means `NEEDS_REVIEW`, not automatic relaxation or F1 tuning.

Calibration saves all allocations, per-step marginal-score distributions, utilization/stop reasons/final marginal scores, and FULL eligibility/consideration/choice/rejection diagnostics by dataset and budget. Counts of FULL attempts are candidate-step events and may revisit a parent. The final pilot analysis also reports these diagnostics.

Three historical notebook-default tests conflict with already completed full-run settings. Their exclusion from this allocator validation was explicitly approved by the user on 2026-11-04. The exact three test IDs and reason are recorded in `calibration/TESTS.json`; original notebooks/tests are unchanged. All other tests must pass.

## Commands

```bash
# Model-free metadata audit
sara_env/bin/python W12-POST/run.py audit
# Tests and allocation-only calibration; NO answer generation
sara_env/bin/python W12-POST/run.py calibrate --execute
# Manual full pilot, after calibration passes
sara_env/bin/python W12-POST/run.py pilot --execute
```

SFR and Qwen use the pinned canonical models sequentially. No automatic model downloads, precision changes or CPU fallback. Compatible missing embeddings are rebuilt under this extension; old SQLite caches are read-only. SciFact retains its full source corpus, so rebuilding missing corpus embeddings can dominate allocation-smoke time.

## Analysis and limitations

Report all dataset/budget cells separately: QA F1/EM; SciFact macro-F1, accuracy, class scores and compliance. Twelve exploratory BEA/control contrasts use paired cluster bootstrap (10,000 resamples, seed42) and an explicitly exploratory Holm family. No pooled cross-task quality score.

Evidence diagnostics include tokens, redundancy, aspect coverage, parent representation, FULL activation and stop behavior. Cosine coverage is a surrogate, not answer support or causal utility. The fixed pilot gate in config controls any optional expanded VAL run; it cannot promote BEA as the final method. There is no separate no-expansion arm, so FULL activation alone cannot establish its causal benefit. See `PRE_PILOT_PATCH_REPORT.md` for actual validation and calibration status.

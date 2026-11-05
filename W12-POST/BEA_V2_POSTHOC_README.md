# Separate BEA_V2 post-hoc engineering evaluation

This is a post-hoc exploratory evaluation at a manually selected engineering operating point (`tau_stop=8e-5`) after the preregistered BEA_V2 calibration failed. It cannot be interpreted as confirmation or as evidence that the original calibration protocol succeeded.

The official `results/post_w12/bea_v2/v3/CALIBRATION_TERMINATED.json` and original notebook remain untouched. Use the **new notebook**, not the terminated one.

## Run the notebook

Open `W12-POST/BEA_QBGF_V2_POSTHOC_TAU8E5.ipynb` using the project's `sara_env` kernel. Change **only `RUN_FULL = True`**, then choose **Run All**. `RUN_ANALYSIS` follows `RUN_FULL` automatically. Keep `RESUME=True`, `OVERWRITE=False`, `AUTO_RUN_MISSING=True`.

1. Preflight verifies the pinned termination result, predictor/fold/schema identities, unchanged research dependencies and exact 50-query-per-dataset development cohort. It lists completed rows, historical controls, missing work and disk space.
2. Preparation copies compatible control rows into separately labelled post-hoc envelopes and constructs missing BEA_V2 evidence with the unchanged allocator at **exactly 0.00008**. Cached source features and predictors are reused; there is no fitting or threshold search.
3. Generation resumes immutable rows and exact-prompt caches. It loads the canonical Qwen BASE_FP16 only when a genuinely missing answer requires it. A free GPU with at least 18 GiB headroom is required. Other processes are never interrupted.
4. Analysis runs only after all 1,200 logical rows exist, then displays tables and both Markdown reports. It does not trigger further runs.

On interruption/shutdown, rerun **Run All** with the same settings. Compatible completed rows are reused; incompatible identities fail loudly. For analysis of a completed run alone, leave `RUN_FULL=False` and set `RUN_ANALYSIS=True`.

There are 300 BEA_V2 answer conditions. The 900 existing control conditions are reused read-only. The preflight should list zero missing controls. If a control is genuinely absent, inspect its key and explicitly set `ALLOW_MISSING_CONTROL_RESTORE=True` to restore only missing controls canonically into this new namespace. Those additional calls are counted separately from the 300 BEA_V2 bound. Corrupted/incompatible rows are rejected, not silently regenerated. Frozen problem caches must be present; restore exact missing source caches rather than substituting retrieval/embeddings.

## Scientific boundaries

- Label: `POST_W12_BEA_V2_POSTHOC_EXPLORATORY`.
- Output: `results/post_w12/bea_v2/posthoc_tau8e5/`.
- Same 50 HotpotQA, 50 QASPER and 50 SciFact development queries; budgets 256/1024.
- Methods: FIXED_SENTENCE, QBGF_FULL, BEA_QBGF_V1, BEA_V2.
- P_help-only utility; P_harm diagnostic-only; eta/lambda, aspects, 1/2/4/8/FULL states, serializer, retriever, prompt, decoding and reader unchanged.
- Eighteen exploratory paired contrasts: BEA_V2 minus each of three controls × three datasets × two budgets. Existing statistics: 10,000 paired cluster bootstrap resamples and cluster label-swap raw p-values, seed 42; Holm across all 18.
- QASPER grouped by paper; SciFact uses the existing cited-document components. No pooled task score. No confirmation or QASPER TEST access.
- Quality-per-token ratios and selection changes are descriptive. This combined-method comparison cannot isolate the causal value of P_help without an additional matched ablation; none is added here.
- No threshold tuning after outcomes, automatic promotion, new attribution or expansion.

## Outputs after you execute

`POSTHOC_PILOT_RESULTS.md`, `POSTHOC_PAIRED_COMPARISONS.csv`, `POSTHOC_EVIDENCE_BEHAVIOR.csv`, `POSTHOC_DECISION.md`, plus quality/cost summary, per-query and aggregate selection changes, utility win/loss diagnostics, immutable selections/results/generation receipts and identity/completion records.

The ten requested questions are covered using all six dataset/budget cells, paired uncertainties, exposure differences, FULL/stopping frequencies and parent/underlying-sentence overlap. No research findings are fabricated before execution.

## Optional manual commands

From the repository root:

```bash
sara_env/bin/python W12-POST/run_bea_v2_posthoc.py status
sara_env/bin/python W12-POST/run_bea_v2_posthoc.py all --execute
```

Optional tests (provided, **not executed** during implementation):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 sara_env/bin/python -m pytest W12-POST/bea_v2_posthoc_tests -q
```

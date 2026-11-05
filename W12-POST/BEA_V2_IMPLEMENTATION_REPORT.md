# BEA_V2 implementation and CPU smoke report

**Implemented and mechanically ready for the manually launched cross-fit development pilot. No 150-query pilot, reader generation, confirmation or unseen expansion was run. V1 was left untouched.**

## Files and execution

- New isolated modules: `src/post_w12/bea_v2/{protocol,data,models,allocator,stages,baselines,analysis,unseen}.py`.
- Notebook: `W12-POST/BEA_QBGF_V2.ipynb`; CLI: `W12-POST/run_bea_v2.py`; configuration: `W12-POST/bea_v2_config.json`; tests: `W12-POST/bea_v2_tests/test_v2.py`.
- Current outputs: **results/post_w12/bea_v2/v2**. Initial V2 preflight artifacts under `bea_v2/v1` are retained rather than overwritten. Historical BEA-v1 remains `bea_qbgf/v3`.
- New derived V2 generation caches are ignored by Git. No historical source, split, freeze, method definition, result or notebook was edited.

## Exact protocol

`U = max(0, P_help − P_harm) × max(0, Rel + 0.5 ΔCoverage − 0.5 Redundancy)`; divide transition value by exact whole-context incremental token cost. The existing BEA-v1 query fragments, SFR pool and serializer are reused. No aspect redesign or semantic-model change.

A global iterative allocator recomputes source transitions after every choice. The next source sentence-count state is 1 → 2 → 4 → 8 → all, with equivalent short-source states deduplicated by exact serialized evidence. Within transitions, candidate ordering is regenerated bottom-up after each hypothetical addition. FULL competes globally after two positive contributions and pays exact entire-source replacement cost. This is **not** an exact static MCKP or learned model of multi-sentence complementarity. FULL can expose zero-weight context; zero direct utility is not an absolute ban on inclusion through FULL.

Two independent L2 logistic models per dataset predict historical helpful/harmful LOO signs. C=1, gamma=1, StandardScaler on training folds only, fixed solver/seed, no class reweighting or hyperparameter search. Probabilities are empirical predictive surrogates, not causal necessity.

## Attribution audit and cross-fitting

9,607 exported historical unit observations survive; all 6,506 unique IDs were reconstructed exactly from content/offset-based source identities. Original prepared/perturbation shards were removed, but the label CSV matches its historical quantitative freeze. All signs and full-minus-ablated scores were verified. **1,070 passage/clause observations were excluded**, never relabeled as sentences.

| dataset   |   observations |   unique_units |   queries |   groups |
|:----------|---------------:|---------------:|----------:|---------:|
| HotpotQA  |           2941 |           1926 |        50 |       50 |
| QASPER    |           3543 |           2164 |        50 |       50 |
| SciFact   |           2053 |           1367 |        50 |       50 |

There are **8,537 usable sentence observations** (555 helpful, 552 harmful, 7,430 neutral), not 8,537 independent queries. Duplicate sentences from different historical intervention contexts are retained and can have different labels. Five GroupKFold splits per dataset use frozen Hotpot query groups, QASPER papers and SciFact cited-document components; every held-out group is absent from its training data. Membership is frozen before fitting.

The fixed 12-feature schema includes query/sentence and passage cosine, retrieval and within-parent rank, normalized source position, sentence/passage token costs, parent sentence count, budget, pool width and static candidate similarity summaries. Optional query-token length is omitted because it is not cached and not needed. Gold/answers/LOO values are never input features. All folds report prevalence, ROC-AUC, average precision, Brier, ten-bin calibration and mean probabilities by actual helpful/neutral/harmful class.

## Held-out predictor diagnostics

Unweighted means across the five held-out folds, descriptive only; repeated units are not independent evidence. PR-AUC is average precision.

| dataset   | target   |   roc_auc |   pr_auc_average_precision |   brier |
|:----------|:---------|----------:|---------------------------:|--------:|
| HotpotQA  | HARMFUL  |    0.3642 |                     0.0518 |  0.0644 |
| HotpotQA  | HELPFUL  |    0.6627 |                     0.1248 |  0.0595 |
| QASPER    | HARMFUL  |    0.5824 |                     0.1212 |  0.0795 |
| QASPER    | HELPFUL  |    0.6647 |                     0.1871 |  0.0816 |
| SciFact   | HARMFUL  |    0.7793 |                     0.1046 |  0.0220 |
| SciFact   | HELPFUL  |    0.8433 |                     0.1169 |  0.0165 |

Hotpot harmfulness ROC-AUC is **0.364**, below chance on average; it is a material warning, not a reason to tune gamma after inspection. Helpful heads show modest discrimination; SciFact has stronger ROC ranks but very rare positives and low absolute precision. Small Brier values alone are uninformative under severe imbalance; constant-prevalence baselines are included in the detailed model report. These diagnostics do not establish that usefulness weighting improves answers.

## Tests and permitted smoke

**18 tests passed**, CPU only. They cover exact historical label signs, grouping/leakage guards, fixed safe features, reproducible logistic probabilities, utility multiplication, harmful/negative-IG suppression, dynamic/sequential recomputation, source states, FULL eligibility/diagnostics, exact token budgets, stopping, predictor-sensitive cache identity and immutable resume, notebook flags, historical source hashes, and a synthetic end-to-end four-method report/plot regression. Scikit-learn emits a nonfatal future deprecation warning for the explicit L2 penalty setting.

The only real allocation smoke used **three unique queries**, one per dataset, at both budgets and six preregistered thresholds (36 cached allocations). It used existing semantic scores plus the CPU tokenizer; SFR and Qwen reader models were not loaded. CPU cross-fit fitting/diagnostics used the historical development labels; no 150-query allocation surface or reader evaluation was executed. The same three smoke queries were rechecked after final logging/identity guard improvements, with no additional query sample or threshold tuning.

The smallest mechanically eligible threshold is **0.00025**. Threshold stopping: **83.3%**; mean utilization **39.3%**; mean cosine-coverage retention relative to zero threshold **98.6%**; empty-evidence fraction **0%**. These are tiny-sample allocation diagnostics, not answer-quality evidence.

| dataset   |   budget |   selected_tokens |   units |   represented_parents |   full_sources | stop_reason        |
|:----------|---------:|------------------:|--------:|----------------------:|---------------:|:-------------------|
| HotpotQA  |      256 |               206 |       7 |                     3 |              1 | LOW_MARGINAL_VALUE |
| HotpotQA  |     1024 |               206 |       7 |                     3 |              1 | LOW_MARGINAL_VALUE |
| QASPER    |      256 |               251 |      11 |                     6 |              0 | BUDGET_EXHAUSTED   |
| QASPER    |     1024 |                16 |       1 |                     1 |              0 | LOW_MARGINAL_VALUE |
| SciFact   |      256 |                79 |       2 |                     1 |              0 | LOW_MARGINAL_VALUE |
| SciFact   |     1024 |                48 |       1 |                     1 |              0 | LOW_MARGINAL_VALUE |

**Under-allocation remains a serious risk:** sampled QASPER@1024 uses 16 tokens and SciFact@1024 uses 48. Budget is a predictor feature, so weights can differ substantially between budgets. No minimum-budget rule or outcome-based retuning was introduced. The preregistered aggregate calibration rule passed; it does not guarantee adequate exposure in each task/budget cell.

## Frozen artifacts, reuse and full notebook

The active BEA_V2 freeze identity is `2de9869af489d5cb887416240946568b08845f7756a1e621f7f918cd21579518`. Predictor weights, feature schema, folds, training sources, package version, calibration, allocator source, original reader/tokenizer/prompt/parser/serializer/decoding and configuration are bound to its identity. Incompatible artifacts fail rather than overwrite.

900 existing control answer rows were verified strictly read-only for reuse. Typical new reader work is at most **300 BEA_V2 answers**, reduced by exact-prompt reuse. Every new row has durable checksums and predictor fold/model identity. Original controls missing later are restored only through canonical historical implementations into V2, not V1. All six task/budget cells and three paired developmental contrasts per cell are implemented (18 group-aware paired tests, 10,000 resamples, seed42); four-method summaries retain task-specific metrics. The preregistered developmental decision is implemented and no expansion is automatic. Bootstrap uncertainty is conditional on fitted cross-fit models, not model-refitting uncertainty.

The final unseen-VAL flag is intentionally **metadata-planning only** at this stage. It excludes all development/training groups, confirmation and any V1-expanded groups. Executing unseen answers requires a separately reviewed all-development predictor/source-preparation extension after the actual pilot decision; cross-fit predictors reject unknown queries. No fresh-generalization claim is available from the already inspected pilot.

When **V1 finishes**, open `W12-POST/BEA_QBGF_V2.ipynb`, select `sara_env`, and **Run All**. Keep default pilot/resume/reuse flags enabled and overwrite/expansion/new attribution disabled. Saved tests, models and smoke/calibration are reused. Full reader inference uses the pinned GPU BASE_FP16 reader; the agent used CPU only and did not touch V1 processes or GPU allocations.

Detailed artifacts: `AUDIT_ATTRIBUTION_DATA.md`, `FEATURE_SCHEMA.json`, `CROSSFIT_FOLDS.json`, `utility_model/UTILITY_MODEL_REPORT.md`, `calibration/ALLOCATION_CALIBRATION_REPORT.md`, and `BEA_V2_FREEZE.json` under the active output root.

Historical source hashes were checked against 1078 recorded immutable inputs; the existing 771-file protected research manifest also remains unchanged. No reader result or full-pilot selection shard exists in V2.

BRANCH = gagan

NO FULL BEA_V2 PILOT WAS STARTED

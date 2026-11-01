# W9G_PRE_FINAL_TRAINING_AUDIT

## A. Audit verdict

**VALID WITH NON-MATERIAL FIXES.** No material target, feature, alignment, preprocessing, fold, threshold, selected-answer, or accounting bug was found. The original nested paper-OOF numerical results remain authoritative **as conditional FIT research results**. No OOF rerun is required, and no features, hyperparameters, model families, folds, grid, epsilon or upstream choices were changed.

This audit added supplementary provenance/reporting and audit-gated final deployment preparation. It did not retrain either study classifier, generate answers, tune outcomes or evaluate DEV/INTERNAL-VAL/TEST. All original A–F and existing G artifacts are unchanged.

## B. Protocol verification

**Target:** for each specific frozen F0 proposal, delta=F1(F0)−F1(D0); ACCEPT_THIS_F0 iff delta>0.01. HELP/NEUTRAL/HURT are >+0.01 / absolute delta<=0.01 / <−0.01. Reconstructed counts: **378 HELP, 1,003 NEUTRAL, 365 HURT**. The binary target is not any-action utility, F2-A utility, oracle K or evidence relevance.

**Actions:** all 1,746 FIT queries/656 papers align to canonical saved D0, unchanged fresh-regeneration F0, E3-SentTopK k3 source text/IDs/chunks, model identity, tokenizer, prompt hashes and greedy 128-token decoding. D0 answer hashes were checked against original frozen D0 rows; F0 answer/prompt/source hashes match saved E3-k3. `action_lineage.csv` records the per-query checks. No D0/F0 regeneration occurred.

**Feature matrix:** exactly the same ordered 32 columns, reconstructed from source and matching saved feature values within 1e-12. Matrix construction has no fitted transformations; outputs depend on whitelisted inference inputs, not labels or row order. No new semantic/confidence/interaction features or subsets were introduced.

**Nested paper OOF:** five outer paper folds; four inner paper folds inside each outer training partition. The saved deterministic fold assignments were reconstructed, including 40 model/outer/inner membership records. Every overlap count in `fold_overlap_counts.csv` is **zero**, and each model has exactly 1,746 unique outer-heldout predictions. There are no missing or duplicate query/model outcomes.

**Preprocessing:** each `fit_model` constructs a fresh pipeline and fits it only on the supplied training rows. Logistic uses median imputation with missing indicators, StandardScaler and fixed logistic regression. Trees use the same imputer and fixed histogram gradient boosting. No global preprocessing precedes splitting. A runtime control-flow test of the frozen training function used synthetic data and non-learning test doubles: **52 fresh pipelines**, actual training-only preprocessing fits, **50 disjoint validation prediction calls**, and no outer-heldout outcome access during inner threshold selection. No actual study classifier was fitted by this test. Historical per-fold pipelines are not archived, so the audit distinguishes frozen-source/runtime-test evidence from direct inspection of unavailable historical fold estimators.

**Estimators:** serialized final pipelines and frozen recipes match: logistic C=1, class_weight=balanced, max_iter=1000, standardized, final seed 42; GBDT 100 iterations, 7 max leaves, depth 3, min leaf 20, L2=1, final seed 42. Inner/outer deterministic seed offsets match the frozen code. sklearn **1.9.1** is recorded and matches the audit environment. There is no synthetic oversampling. Class balance affects logistic fitting only; downstream F1 remains an equally weighted query mean. Actual inner/outer training partitions all contain both classes, so no single-class fallback was used.

### Threshold audit

The exact frozen grid is `{0,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90,.95,1,1.01}`. Selection maximizes inner-OOF selected-answer F1; within 0.003, it prefers lower HURT, lower ACCEPT, lower candidate cost, closest to 0.5, then higher threshold. Source and independently reconstructed saved selections agree.

Per-fold selections below use **only that outer fold's training partition**. Every stored outer decision equals its saved probability compared with the corresponding selected threshold. Eligible grids and inner best F1 are exported without changing decisions:

| model                  |   outer_fold |   selected_threshold |   inner_best_f1 | eligible_thresholds                             |     hurt |   acceptance_rate |
|:-----------------------|-------------:|---------------------:|----------------:|:------------------------------------------------|---------:|------------------:|
| logistic               |            0 |             0.700000 |        0.387228 | [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.7] | 0.075215 |          0.218481 |
| logistic               |            1 |             0.700000 |        0.396694 | [0.6, 0.7]                                      | 0.067287 |          0.222620 |
| logistic               |            2 |             0.700000 |        0.391234 | [0.7]                                           | 0.074445 |          0.236220 |
| logistic               |            3 |             1.000000 |        0.395201 | [0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.01]           | 0.000000 |          0.000000 |
| logistic               |            4 |             0.700000 |        0.397927 | [0.6, 0.7]                                      | 0.075877 |          0.236220 |
| hist_gradient_boosting |            0 |             0.400000 |        0.386240 | [0.0, 0.05, 0.1, 0.15, 0.3, 0.4]                | 0.090258 |          0.258596 |
| hist_gradient_boosting |            1 |             0.400000 |        0.392483 | [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]           | 0.084467 |          0.233357 |
| hist_gradient_boosting |            2 |             0.400000 |        0.391126 | [0.15, 0.2, 0.4]                                | 0.087330 |          0.258411 |
| hist_gradient_boosting |            3 |             0.800000 |        0.397724 | [0.3, 0.7, 0.8]                                 | 0.000000 |          0.002147 |
| hist_gradient_boosting |            4 |             0.300000 |        0.396318 | [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]                | 0.134574 |          0.394417 |

**Honest OOF versus deployment:** logistic uses outer thresholds **[0.70,0.70,0.70,1.00,0.70]**, not a global 0.70. GBDT uses **[0.40,0.40,0.40,0.80,0.30]**. Pooled outer-OOF deployment selection is stored separately at outer_fold=-1 and correctly marked post-selection. It chooses logistic **0.70** and GBDT **0.40**. Applying pooled logistic 0.70 retrospectively gives **0.396717 F1** on those probabilities; this is **not** the honest nested-OOF result of **0.389443**. Original summaries retain the correct nested result; no numerical relabeling was needed.

Raw inner OOF probabilities were not archived. Their grid cannot be independently rescored without retraining, which is neither necessary nor permitted absent a bug. Threshold provenance is instead checked through frozen source, restricted-access synthetic traces, saved fold memberships, grid sample counts, exact selection reconstruction and outer decision reproduction. This is an explicit evidence limit, not a claim that historical intermediate arrays were inspected.

## C. Leakage and missing-value audit

Every feature is available **after** the specific F0 candidate is generated. IDs remain metadata. The matrix rejects extra/forbidden columns, including gold/F1/delta/HELP/HURT/ACCEPT/oracle, paper or query IDs, row order, selected answer and threshold results. Perturbing outcome fields leaves features unchanged; reversing input dictionary order leaves sorted metadata/features unchanged. No gold-derived answer types or held-out fold aggregates enter the matrix.

E3-k3 and F choices were previously selected globally on FIT. The verifier OOF estimate is therefore **conditional on frozen upstream development**, not independent end-to-end confirmation. That known limitation does not constitute a new verifier feature-leakage bug.

| feature                                 | group      | inference available?   | source                                         | leakage risk                          | verdict   |
|:----------------------------------------|:-----------|:-----------------------|:-----------------------------------------------|:--------------------------------------|:----------|
| normalized_equal                        | V0         | Yes, post-generation   | normalized tokens of y0 and specific y1        | No outcome/ID input                   | PASS      |
| token_edit_distance                     | V0         | Yes, post-generation   | Levenshtein(y0 tokens,y1 tokens)               | No outcome/ID input                   | PASS      |
| token_jaccard                           | V0         | Yes, post-generation   | token-set overlap of y0/y1                     | No outcome/ID input                   | PASS      |
| added_token_count                       | V0         | Yes, post-generation   | positive multiset y1-minus-y0                  | No outcome/ID input                   | PASS      |
| removed_token_count                     | V0         | Yes, post-generation   | positive multiset y0-minus-y1                  | No outcome/ID input                   | PASS      |
| initial_answer_length                   | V0         | Yes, post-generation   | y0 token count                                 | No outcome/ID input                   | PASS      |
| candidate_answer_length                 | V0         | Yes, post-generation   | y1 token count                                 | No outcome/ID input                   | PASS      |
| answer_length_difference                | V0         | Yes, post-generation   | len(y1)-len(y0)                                | No outcome/ID input                   | PASS      |
| answer_length_ratio                     | V0         | Yes, post-generation   | len(y1)/max(len(y0),1)                         | No outcome/ID input                   | PASS      |
| changed_fraction                        | V0         | Yes, post-generation   | edit distance/max answer length                | No outcome/ID input                   | PASS      |
| one_answer_substring                    | V0         | Yes, post-generation   | normalized y0/y1 substring relationship        | No outcome/ID input                   | PASS      |
| initial_evidence_overlap                | V1_lexical | Yes, post-generation   | lexical y0/E overlap                           | No outcome/ID input                   | PASS      |
| candidate_evidence_overlap              | V1_lexical | Yes, post-generation   | lexical y1/E overlap                           | No outcome/ID input                   | PASS      |
| evidence_overlap_delta                  | V1_lexical | Yes, post-generation   | candidate minus initial lexical overlap        | No outcome/ID input                   | PASS      |
| added_tokens_in_evidence_fraction       | V1_lexical | Yes, post-generation   | added y1 tokens present in E                   | No outcome/ID input                   | PASS      |
| removed_tokens_absent_evidence_fraction | V1_lexical | Yes, post-generation   | removed y0 tokens absent from E                | No outcome/ID input                   | PASS      |
| initial_generated_tokens                | V2_length  | Yes, post-generation   | saved first-pass generation length             | No outcome/ID input                   | PASS      |
| candidate_generated_tokens              | V2_length  | Yes, post-generation   | saved F0 generation length                     | No outcome/ID input                   | PASS      |
| top_sentence_cosine                     | V3         | Yes, post-generation   | first frozen E3 selected cosine                | Conditional frozen upstream selection | PASS      |
| second_sentence_cosine                  | V3         | Yes, post-generation   | second frozen E3 selected cosine               | Conditional frozen upstream selection | PASS      |
| third_sentence_cosine                   | V3         | Yes, post-generation   | third frozen E3 selected cosine                | Conditional frozen upstream selection | PASS      |
| mean_selected_cosine                    | V3         | Yes, post-generation   | saved E3 selected cosine mean                  | Conditional frozen upstream selection | PASS      |
| min_selected_cosine                     | V3         | Yes, post-generation   | saved E3 selected cosine minimum               | Conditional frozen upstream selection | PASS      |
| next_unselected_cosine                  | V3         | Yes, post-generation   | saved E3 next-candidate cosine                 | Conditional frozen upstream selection | PASS      |
| selected_vs_next_margin                 | V3         | Yes, post-generation   | saved E3 minimum-selected minus next cosine    | Conditional frozen upstream selection | PASS      |
| evidence_tokens                         | V3         | Yes, post-generation   | frozen E3 evidence length                      | Conditional frozen upstream selection | PASS      |
| mean_source_chunk_rank                  | V3         | Yes, post-generation   | mean document-order selected chunk rank; no ID | Conditional frozen upstream selection | PASS      |
| selected_chunk_count                    | V3         | Yes, post-generation   | number of selected source chunks               | Conditional frozen upstream selection | PASS      |
| query_tokens                            | V4         | Yes, post-generation   | question token count                           | No outcome/ID input                   | PASS      |
| question_mark                           | V4         | Yes, post-generation   | question terminal punctuation                  | No outcome/ID input                   | PASS      |
| starts_wh_question                      | V4         | Yes, post-generation   | fixed generic opening word set                 | No outcome/ID input                   | PASS      |
| starts_auxiliary_question               | V4         | Yes, post-generation   | fixed generic auxiliary opening word set       | No outcome/ID input                   | PASS      |

### Missingness for all 32 features

All current FIT feature missing counts are zero, including next-unselected cosine, ranking metadata and generation lengths. Actual zeros are valid observed indicators/counts/overlap values. The source retains unavailable rank/length measurements as NaN; it does not globally replace missing observations with zero. Any imputer/scaler fitting remains inside each training partition. Optional logprobs/entropy/EOS and semantic-answer features are **absent**, not observed zeros.

| feature                                 |   missing_count |   missing_fraction | imputation                               |
|:----------------------------------------|----------------:|-------------------:|:-----------------------------------------|
| normalized_equal                        |               0 |           0.000000 | Median + indicators, training folds only |
| token_edit_distance                     |               0 |           0.000000 | Median + indicators, training folds only |
| token_jaccard                           |               0 |           0.000000 | Median + indicators, training folds only |
| added_token_count                       |               0 |           0.000000 | Median + indicators, training folds only |
| removed_token_count                     |               0 |           0.000000 | Median + indicators, training folds only |
| initial_answer_length                   |               0 |           0.000000 | Median + indicators, training folds only |
| candidate_answer_length                 |               0 |           0.000000 | Median + indicators, training folds only |
| answer_length_difference                |               0 |           0.000000 | Median + indicators, training folds only |
| answer_length_ratio                     |               0 |           0.000000 | Median + indicators, training folds only |
| changed_fraction                        |               0 |           0.000000 | Median + indicators, training folds only |
| one_answer_substring                    |               0 |           0.000000 | Median + indicators, training folds only |
| initial_evidence_overlap                |               0 |           0.000000 | Median + indicators, training folds only |
| candidate_evidence_overlap              |               0 |           0.000000 | Median + indicators, training folds only |
| evidence_overlap_delta                  |               0 |           0.000000 | Median + indicators, training folds only |
| added_tokens_in_evidence_fraction       |               0 |           0.000000 | Median + indicators, training folds only |
| removed_tokens_absent_evidence_fraction |               0 |           0.000000 | Median + indicators, training folds only |
| initial_generated_tokens                |               0 |           0.000000 | Median + indicators, training folds only |
| candidate_generated_tokens              |               0 |           0.000000 | Median + indicators, training folds only |
| top_sentence_cosine                     |               0 |           0.000000 | Median + indicators, training folds only |
| second_sentence_cosine                  |               0 |           0.000000 | Median + indicators, training folds only |
| third_sentence_cosine                   |               0 |           0.000000 | Median + indicators, training folds only |
| mean_selected_cosine                    |               0 |           0.000000 | Median + indicators, training folds only |
| min_selected_cosine                     |               0 |           0.000000 | Median + indicators, training folds only |
| next_unselected_cosine                  |               0 |           0.000000 | Median + indicators, training folds only |
| selected_vs_next_margin                 |               0 |           0.000000 | Median + indicators, training folds only |
| evidence_tokens                         |               0 |           0.000000 | Median + indicators, training folds only |
| mean_source_chunk_rank                  |               0 |           0.000000 | Median + indicators, training folds only |
| selected_chunk_count                    |               0 |           0.000000 | Median + indicators, training folds only |
| query_tokens                            |               0 |           0.000000 | Median + indicators, training folds only |
| question_mark                           |               0 |           0.000000 | Median + indicators, training folds only |
| starts_wh_question                      |               0 |           0.000000 | Median + indicators, training folds only |
| starts_auxiliary_question               |               0 |           0.000000 | Median + indicators, training folds only |

## D. Independent OOF metric reconstruction

Selected answer strings, F1, EM, HELP/HURT, action deltas, costs and decisions were reconstructed directly from saved outer-heldout probabilities/decisions and frozen action rows. Both model summaries match within 1e-12. No final-model training-set prediction was evaluated.

| model                  |   n_queries |       f1 |   exact_match |   acceptance_rate |     help |     hurt |   mean_selected_action_delta |   harm_rejection_rate |   help_retention_rate |   false_accept_rate |
|:-----------------------|------------:|---------:|--------------:|------------------:|---------:|---------:|-----------------------------:|----------------------:|----------------------:|--------------------:|
| logistic               |        1746 | 0.389443 |      0.228522 |          0.183849 | 0.074456 | 0.062428 |                     0.003812 |              0.701370 |              0.343915 |            0.595016 |
| hist_gradient_boosting |        1746 | 0.389266 |      0.228522 |          0.223368 | 0.087629 | 0.079038 |                     0.003635 |              0.621918 |              0.404762 |            0.607692 |

Logistic accepts **321 proposals**: **130 HELP, 109 HURT, 82 NEUTRAL**. It rejects **256/365 HURT** proposals, retains **130/378 HELP**, and misses **248/378 HELP**. False-accept rate is **191/321=59.50%**, including neutral acceptance; harmful-accept fraction is **109/321=33.96%**. These are descriptive action-specific rates, not classifier accuracy or evidence-causality claims.

### Critical reference table

| policy                    |       f1 |     help |     hurt |   lm_calls |   cumulative_prompt |
|:--------------------------|---------:|---------:|---------:|-----------:|--------------------:|
| D0                        | 0.385631 | 0.000000 | 0.000000 |   1.000000 |          940.544101 |
| F0                        | 0.390024 | 0.216495 | 0.209049 |   2.000000 |         2026.312715 |
| F2-A                      | 0.389616 | 0.021764 | 0.008591 |   2.000000 |         2119.990263 |
| logistic G2               | 0.389443 | 0.074456 | 0.062428 |   2.000000 |         2026.312715 |
| hist_gradient_boosting G2 | 0.389266 | 0.087629 | 0.079038 |   2.000000 |         2026.312715 |

F0 has substantial HELP and HURT; F2-A has very low HURT but low HELP retention. Logistic is an intermediate tradeoff. It does **not** dominate F2-A, which has slightly higher point F1 and much lower HURT.

## E. Independent paper-bootstrap reconstruction

The independent implementation resamples **whole paper clusters**, 10,000 times, seed 42, and weights each resampled paper by its included query count. Logistic−D0 and Logistic−F0 intervals match the original saved values. Logistic−F2-A is an added **post-hoc descriptive comparison**, not preregistered superiority testing.

| comparison      |   point_estimate |   ci95_low |   ci95_high |   papers |   resamples |   seed | preregistered_comparison   |
|:----------------|-----------------:|-----------:|------------:|---------:|------------:|-------:|:---------------------------|
| Logistic - D0   |         0.003812 |  -0.005019 |    0.012492 |      656 |       10000 |     42 | True                       |
| Logistic - F0   |        -0.000581 |  -0.010087 |    0.008749 |      656 |       10000 |     42 | True                       |
| Logistic - F2-A |        -0.000173 |  -0.008565 |    0.008058 |      656 |       10000 |     42 | False                      |

Every interval crosses zero. These are conditional, unadjusted intervals for the fixed selected-model outputs; they do **not** account for the model-selection search or global upstream development and must not be described as independent confirmation. Individual model OOF scores, FIT-selected logistic and the final all-FIT estimator remain distinct.

## F. Cost and runtime audit

Every KEEP and ACCEPT row retains the **entire saved F0 candidate pipeline cost**, averaging **2026.312715 cumulative prompt tokens and 2 LM calls**. No D0-only refund is applied on rejected proposals. Preflight oracle chosen-action costs remain retrospective and explicitly exclude searching/generating all candidates. A post-generation verifier cannot achieve those oracle-style cost savings.

Saved F0 answer latency is inherited (mean approximately 1.022722 s); verifier prediction latency is measured per outer batch then amortized per query (logistic mean approximately 0.0000171 s). Initial-draft generation, sentence-index/query-embedding work remain cached upstream components. **Feature computation and preprocessing stage latency were not separately measured** and are now explicitly marked unknown. No end-to-end latency is claimed or synthesized by adding incompatible cached times.

## G. Issues and non-material fixes

| Issue | Classification | Action |
| --- | --- | --- |
| Selected per-fold thresholds were implicit in complete grid tables | Non-material metadata | Exported `selected_thresholds.csv`: chosen threshold, inner best F1, eligible grid, HURT and acceptance. No decisions changed. |
| OOF runner already fits final full-FIT estimators inside the OOF directory; packaging could be confused with validation | Documentation/metadata | Clarified that archived `logistic.joblib` is already all-FIT. Added separate audit-gated final packaging/refit code; no original artifact overwritten. |
| Raw inner probabilities/fold estimators and separate feature-computation timing were not archived | Non-material provenance/measurement limitation | Explicitly report what was verifiable and what remains unavailable; no invented values and no unnecessary rerun. |
| Previously published implementation notes describe their pre-training snapshot | Documentation only | This audit records current completed training state; historical scientific artifacts and reports remain unchanged. |

**Material issues found: none.** No existing controller/training/feature/config code was changed. Audit helpers, tests, supplementary metadata and final export preparation live separately.

## H. Is an OOF rerun required?

**No.** The frozen 32-feature, two-classifier nested paper-OOF study remains the authoritative conditional FIT estimate. Do not add features, retune class weights/grid/hyperparameters, alter epsilon/folds, reopen E3-k or F prompts, or reinterpret pooled-threshold performance as nested OOF.

## I. Is final full-FIT logistic use safe?

**Technically safe under this valid audit, with no superiority claim.** The archived logistic pipeline already has all-FIT medians/scaler statistics, exactly 32 input columns, frozen parameters and scaler sample count **1,746**. Frozen source fits final estimators after OOF outcomes are collected and does not use their predictions for OOF summaries. Its labels/query membership are supported by frozen code/data lineage, not a retrospective claim that an original per-fit execution log was archived.

A separate `final_fit/` package is prepared for later held-out evaluation/W9-H. It will contain the full pipeline, classifier, preprocessing pipeline, schema, threshold=0.70, training query/paper ID hashes, sklearn/code/config/upstream hashes and manifest. It refuses invalid audits, changed inputs/schema and existing or OOF output paths. It never reports final-model training-set performance. **No final package or real fit was launched by this audit.**

## J. Exact final deployment commands

From the repository root, the preferred action avoids redundant training by exporting the already-audited full-FIT pipeline:

```bash
./sara_env/bin/python -m src.week9.action_verification_g_audit.final_fit --audit results/week9/w9g_action_verification/v1/audit_pre_final/audit_status.json --reuse-audited-model
```

If an explicit fresh final-only fit is desired instead, the prepared command uses **exactly** the same 32 features, preprocessing, logistic parameters and selected threshold, without rerunning OOF:

```bash
./sara_env/bin/python -m src.week9.action_verification_g_audit.final_fit --audit results/week9/w9g_action_verification/v1/audit_pre_final/audit_status.json --refit
```

Neither command was run here. They are mutually exclusive; no action is the default. Both write only `results/week9/w9g_action_verification/v1/final_fit/` and refuse to overwrite an existing package. Do not rerun the G notebook's controller-training flag simply to obtain a final model: the bounded OOF study is complete.

The audit itself is reproducible with:

```bash
./sara_env/bin/python -m src.week9.action_verification_g_audit.audit
```

## K. What remains frozen and validation performed

Freeze the exact D0/F0 tuples, E3-k3 text/IDs/chunks/model/prompt/config hashes; 32-feature definitions/order; target epsilon 0.01; fold assignments; preprocessing recipes; logistic/GBDT parameters and deterministic seeds; threshold grid/tie rule; selected logistic; pooled deployment threshold 0.70; existing OOF outputs and their post-selection interpretation. Optional semantic/confidence features or another model would require a **future preregistered follow-up**, not a repair to this experiment.

Audit tests: **37 passed in 16.86s**. Broader Week-9 checks: **209 passed, 5 deselected in 24.60s**. Tests cover the requested counts, 32-column whitelist, source/action hashes, fold/threshold isolation, actual missingness, frozen models, selected answers/HELP/HURT, KEEP cost, independent paper bootstrap, valid-audit final-fit gating, exact final schema, refusal to overwrite OOF, explicit action flags, no final training-set prediction path and branch gagan. Existing notebook-state assertions are excluded from the broad run because saved, completed notebooks have user-enabled run flags/changed outputs; those historical artifacts are preserved.

The five excluded notebook-state tests are `notebooks_have_safe_defaults_and_no_test_dropdown`, `e_notebook_disabled_path_executes_without_generation`, `old_notebook_prefix_preserved_and_all_flags_disabled`, `notebook_all_disabled_executes_without_generation` and `notebook_default_preflight_only`. The completed F/G notebooks retain the user's enabled run flags; earlier E/PRE notebook outputs/default conventions also differ from historical test snapshots. No numerical audit assertion was excluded.

Targeted test command: `./sara_env/bin/python -m pytest tests/week9/test_g_pre_final_audit.py -q`.

The synthetic runtime trace fitted actual preprocessing only on synthetic training partitions and used constant non-learning classifier test doubles. It is not a new model or a rerun of the study. Independent numerical audit files and source/data hash seals are under `results/week9/w9g_action_verification/v1/audit_pre_final/`; test logs are under `results/week9/patch_verification/w9g_pre_final_audit/`. Original scientific artifacts were rechecked unchanged after the audit.

BRANCH = gagan

AUDIT STATUS = VALID WITH NON-MATERIAL FIXES

NO NEW FEATURES OR MODELS WERE INTRODUCED

TEST WAS NOT TOUCHED

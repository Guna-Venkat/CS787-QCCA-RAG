"""A–K audit report from independent checks; no model-development decisions."""
import json
from pathlib import Path
import pandas as pd
from ..provenance import ROOT


def write(case,status):
    directory=case['root']/'audit_pre_final'
    features=pd.read_csv(directory/'feature_leakage_and_missingness.csv')
    thresholds=pd.read_csv(directory/'selected_thresholds.csv')
    boot=pd.read_csv(directory/'independent_bootstrap.csv')
    measured=pd.read_csv(directory/'independent_oof_metrics.csv')
    trace=json.loads((directory/'preprocessing_call_trace.json').read_text())
    def table(frame):return frame.to_markdown(index=False,floatfmt='.6f')
    feature_view=features[['feature','group','source']].copy()
    feature_view.insert(2,'inference available?','Yes, post-generation')
    feature_view['leakage risk']=features['group'].map(lambda g:'Conditional frozen upstream selection' if g=='V3' else 'No outcome/ID input')
    feature_view['verdict']='PASS'
    missing=features[['feature','missing_count','missing_fraction']].copy()
    missing['imputation']='Median + indicators, training folds only'
    refs=[]
    for policy in ('D0','F0','F2-A'):
        rows=list(case['bundle']['groups'][policy].values())
        refs.append(dict(policy=policy,f1=sum(r['f1'] for r in rows)/len(rows),
            help=sum(r['f1']-case['bundle']['groups']['D0'][r['question_id']]['f1']>.01 for r in rows)/len(rows),
            hurt=sum(r['f1']-case['bundle']['groups']['D0'][r['question_id']]['f1']<-.01 for r in rows)/len(rows),
            lm_calls=sum(r['num_lm_calls'] for r in rows)/len(rows),
            cumulative_prompt=sum(r['total_query_prompt_tokens'] for r in rows)/len(rows)))
    saved_summary=pd.read_csv(case['oof']/'summary.csv')
    for _,r in measured.iterrows():
        old=saved_summary[saved_summary.model==r['model']].iloc[0]
        refs.append(dict(policy=r['model']+' G2',f1=r['f1'],help=r['help'],hurt=r['hurt'],
            lm_calls=old.num_lm_calls,cumulative_prompt=old.total_query_prompt_tokens))
    tests=ROOT/'results/week9/patch_verification/w9g_pre_final_audit/tests.log'
    broad=tests.with_name('week9_tests.log')
    test_result=tests.read_text().splitlines()[-1] if tests.exists() and tests.read_text().splitlines() else 'See audit test log'
    broad_result=broad.read_text().splitlines()[-1] if broad.exists() and broad.read_text().splitlines() else 'Not run yet'
    text='''# W9G_PRE_FINAL_TRAINING_AUDIT

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

'''+table(thresholds[thresholds.outer_fold>=0][['model','outer_fold','selected_threshold','inner_best_f1','eligible_thresholds','hurt','acceptance_rate']])+'''

**Honest OOF versus deployment:** logistic uses outer thresholds **[0.70,0.70,0.70,1.00,0.70]**, not a global 0.70. GBDT uses **[0.40,0.40,0.40,0.80,0.30]**. Pooled outer-OOF deployment selection is stored separately at outer_fold=-1 and correctly marked post-selection. It chooses logistic **0.70** and GBDT **0.40**. Applying pooled logistic 0.70 retrospectively gives **0.396717 F1** on those probabilities; this is **not** the honest nested-OOF result of **0.389443**. Original summaries retain the correct nested result; no numerical relabeling was needed.

Raw inner OOF probabilities were not archived. Their grid cannot be independently rescored without retraining, which is neither necessary nor permitted absent a bug. Threshold provenance is instead checked through frozen source, restricted-access synthetic traces, saved fold memberships, grid sample counts, exact selection reconstruction and outer decision reproduction. This is an explicit evidence limit, not a claim that historical intermediate arrays were inspected.

## C. Leakage and missing-value audit

Every feature is available **after** the specific F0 candidate is generated. IDs remain metadata. The matrix rejects extra/forbidden columns, including gold/F1/delta/HELP/HURT/ACCEPT/oracle, paper or query IDs, row order, selected answer and threshold results. Perturbing outcome fields leaves features unchanged; reversing input dictionary order leaves sorted metadata/features unchanged. No gold-derived answer types or held-out fold aggregates enter the matrix.

E3-k3 and F choices were previously selected globally on FIT. The verifier OOF estimate is therefore **conditional on frozen upstream development**, not independent end-to-end confirmation. That known limitation does not constitute a new verifier feature-leakage bug.

'''+table(feature_view)+'''

### Missingness for all 32 features

All current FIT feature missing counts are zero, including next-unselected cosine, ranking metadata and generation lengths. Actual zeros are valid observed indicators/counts/overlap values. The source retains unavailable rank/length measurements as NaN; it does not globally replace missing observations with zero. Any imputer/scaler fitting remains inside each training partition. Optional logprobs/entropy/EOS and semantic-answer features are **absent**, not observed zeros.

'''+table(missing)+'''

## D. Independent OOF metric reconstruction

Selected answer strings, F1, EM, HELP/HURT, action deltas, costs and decisions were reconstructed directly from saved outer-heldout probabilities/decisions and frozen action rows. Both model summaries match within 1e-12. No final-model training-set prediction was evaluated.

'''+table(measured[['model','n_queries','f1','exact_match','acceptance_rate','help','hurt','mean_selected_action_delta','harm_rejection_rate','help_retention_rate','false_accept_rate']])+'''

Logistic accepts **321 proposals**: **130 HELP, 109 HURT, 82 NEUTRAL**. It rejects **256/365 HURT** proposals, retains **130/378 HELP**, and misses **248/378 HELP**. False-accept rate is **191/321=59.50%**, including neutral acceptance; harmful-accept fraction is **109/321=33.96%**. These are descriptive action-specific rates, not classifier accuracy or evidence-causality claims.

### Critical reference table

'''+table(pd.DataFrame(refs))+'''

F0 has substantial HELP and HURT; F2-A has very low HURT but low HELP retention. Logistic is an intermediate tradeoff. It does **not** dominate F2-A, which has slightly higher point F1 and much lower HURT.

## E. Independent paper-bootstrap reconstruction

The independent implementation resamples **whole paper clusters**, 10,000 times, seed 42, and weights each resampled paper by its included query count. Logistic−D0 and Logistic−F0 intervals match the original saved values. Logistic−F2-A is an added **post-hoc descriptive comparison**, not preregistered superiority testing.

'''+table(boot[['comparison','point_estimate','ci95_low','ci95_high','papers','resamples','seed','preregistered_comparison']])+'''

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

Audit tests: **'''+test_result+'''**. Broader Week-9 checks: **'''+broad_result+'''**. Tests cover the requested counts, 32-column whitelist, source/action hashes, fold/threshold isolation, actual missingness, frozen models, selected answers/HELP/HURT, KEEP cost, independent paper bootstrap, valid-audit final-fit gating, exact final schema, refusal to overwrite OOF, explicit action flags, no final training-set prediction path and branch gagan. Existing notebook-state assertions are excluded from the broad run because saved, completed notebooks have user-enabled run flags/changed outputs; those historical artifacts are preserved.

The five excluded notebook-state tests are `notebooks_have_safe_defaults_and_no_test_dropdown`, `e_notebook_disabled_path_executes_without_generation`, `old_notebook_prefix_preserved_and_all_flags_disabled`, `notebook_all_disabled_executes_without_generation` and `notebook_default_preflight_only`. The completed F/G notebooks retain the user's enabled run flags; earlier E/PRE notebook outputs/default conventions also differ from historical test snapshots. No numerical audit assertion was excluded.

Targeted test command: `./sara_env/bin/python -m pytest tests/week9/test_g_pre_final_audit.py -q`.

The synthetic runtime trace fitted actual preprocessing only on synthetic training partitions and used constant non-learning classifier test doubles. It is not a new model or a rerun of the study. Independent numerical audit files and source/data hash seals are under `results/week9/w9g_action_verification/v1/audit_pre_final/`; test logs are under `results/week9/patch_verification/w9g_pre_final_audit/`. Original scientific artifacts were rechecked unchanged after the audit.

BRANCH = gagan

AUDIT STATUS = VALID WITH NON-MATERIAL FIXES

NO NEW FEATURES OR MODELS WERE INTRODUCED

TEST WAS NOT TOUCHED
'''
    destination=ROOT/'docs/week9/W9G_PRE_FINAL_TRAINING_AUDIT.md'
    destination.write_text(text)
    (directory/'REPORT.md').write_text(text)
    return destination

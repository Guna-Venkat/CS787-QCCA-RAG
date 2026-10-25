# W9-G-PRE — Action-specific verification: implementation and results

**The saved-output preflight is complete and recommends G2: KEEP_D0 vs ACCEPT_THIS_F0.** No LM generation, embedding-model loading or controller training was performed. All completed A–E3/F artifacts and historical W9-G code/results are preserved. Work is on `gagan`.

## 1. Scientific question and version isolation

W9-G now asks whether inference-available information about the **specific** tuple `(question, original answer y0, frozen E3-k3 evidence E, proposed F0 answer y1)` can distinguish a beneficial answer change from a harmful one. It does not predict global K or decide whether to retrieve without observing the action.

The new experiment lives under `results/week9/w9g_action_verification/v1/`. The existing `adaptive_ke.py`, `w9g.yaml`, historical G notebook, scripts and result hierarchy are not replaced. The new notebook is `notebooks/week9/week9g_action_verification.ipynb`.

Frozen evidence remains FIXED-5 / SEMANTIC@20 / DraftSemantic coarse K=2 / query-only canonical MiniLM ranking / E3 top-3 exact sentences / source-order restoration. No retrieval, selection, prompt or generation changes are made.

## 2. Read-only input/handoff validation

The compatibility loader verifies completed W9-F FIT coverage and protocol/config/prompt/source fingerprints, immutable F2/intervention freezes, saved result checksums, full E3 protocol/k3 freeze and artifact hashes, source sentence IDs/text hashes, question/paper identities, original D0 draft/score and F0 equivalence to E3-k3. It independently aligns every requested policy to **1,746 FIT queries across 656 papers**. Missing, duplicate or misaligned actions fail loudly.

**Important distinction:** the frozen F2 prompt is **F2-A**, while the separate overall W9-F intervention freeze selected **F0** under the existing cost-first rule. The F handoff therefore contains F0 scores. G preserves both original freezes and loads F2-A from the complete saved F result rows to construct PRE-A/C; it does not reinterpret or rewrite the handoff.

Gold-derived saved FIT F1/EM and deltas are analysis/supervision only. No raw gold answer text is needed by this preflight, no model is loaded, and no DEV/INTERNAL-VAL/TEST split is used for selection.

## 3. Saved fixed actions

| Action | F1 | EM | Mean cumulative prompt tokens | Conceptual LM calls |
| --- | ---: | ---: | ---: | ---: |
| D0: keep original draft | 0.385631 | 0.227950 | 940.54 | 1 |
| F0: fresh E3-k3 regeneration | 0.390024 | 0.226231 | 2026.31 | 2 |
| F2-A: frozen evidence-anchored revision | 0.389616 | 0.231386 | 2119.99 | 2 |

F0 creates many useful and harmful changes. F2-A preserves D0 on **94.39%** of queries and changes the gain/harm tradeoff. The question is whether a verifier can accept the useful F0 changes without accepting its harmful ones.

## 4. Exact action spaces and tie rules

- **PRE-A:** `{D0, F2-A}` — remaining value around the frozen conservative action.
- **PRE-B:** `{D0, F0}` — ceiling for verifying the actual aggressive F0 proposal.
- **PRE-C:** `{D0, F0, F2-A}` — diagnostic of extra action diversity.

Exact oracle: choose maximum per-query saved F1; exact ties prefer **D0 -> F2-A -> F0**, restricted to the action space. Tie diagnostics use exact equality, not approximate matching.

Epsilon oracle: **epsilon=0.01**, choose the first action in **D0 -> F2-A -> F0** whose F1 is within epsilon of the maximum. This favors D0 on near-ties and F2-A over F0 when both interventions qualify. It is a safety-oriented preference, not a claim that F2-A has lower prompt cost. Epsilon choices need not maximize F1 exactly, and adding an eligible action can change the preferred score.

Headroom is always oracle mean F1 minus the **highest-F1 fixed action in that action space**, not automatically minus D0.

## 5. PRE-A/B/C results

| Space | Best fixed F1 | Exact oracle F1 | Exact headroom | Epsilon oracle F1 | Epsilon headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| PRE-A: D0/F2-A | 0.389616 | 0.393462 | 0.003846 | 0.393449 | 0.003833 |
| PRE-B: D0/F0 | 0.390024 | 0.453880 | 0.063856 | 0.453843 | 0.063819 |
| PRE-C: D0/F0/F2-A | 0.390024 | 0.454843 | 0.064819 | 0.454801 | 0.064777 |

PRE-B's epsilon headroom is **0.063819 raw F1**, or **6.382 percentage points**, versus PRE-A's **0.383 percentage points**. The descriptive verifier advantage is **0.059986 F1**. These are optimistic retrospective ceilings; they do not establish learnable signal or a deployable improvement.

### Action distributions

| Space/oracle | D0 count (%) | F0 count (%) | F2-A count (%) |
| --- | ---: | ---: | ---: |
| PRE-A exact | 1704 (97.59%) | — | 42 (2.41%) |
| PRE-A epsilon | 1708 (97.82%) | — | 38 (2.18%) |
| PRE-B exact | 1358 (77.78%) | 388 (22.22%) | — |
| PRE-B epsilon | 1368 (78.35%) | 378 (21.65%) | — |
| PRE-C exact | 1349 (77.26%) | 366 (20.96%) | 31 (1.78%) |
| PRE-C epsilon | 1360 (77.89%) | 357 (20.45%) | 29 (1.66%) |

`oracle_distribution.csv` also breaks selected actions into HELP/HURT/NEUTRAL versus D0. Oracle-selected HURT is zero by construction because D0 is always available; this is not evidence that a learned verifier can achieve zero HURT.

## 6. Does F2-A add unique useful actions?

| Mutually exclusive exact top-score category | Queries | Fraction |
| --- | ---: | ---: |
| D0 uniquely best | 15 | 0.86% |
| F0 uniquely best | 366 | 20.96% |
| F2-A uniquely best | 13 | 0.74% |
| Top tie D0/F0 only | 0 | 0.00% |
| Top tie D0/F2-A only | 358 | 20.50% |
| Top tie F0/F2-A only | 18 | 1.03% |
| Three-way top tie | 976 | 55.90% |

- F2-A's extra **exact** oracle value beyond D0/F0: **+0.000963 F1**.
- Extra **epsilon** oracle value: **+0.000958 F1**.
- F2-A is uniquely exact-best on **13 queries (0.74%)**.
- It is the **only epsilon-eligible action** on **9 queries (0.52%)**.
- F2-A exactly preserves D0 text on **94.39%**; its F1 equals D0 on **96.74%**.

F2-A adds little unique oracle value on this FIT surface. This supports binary verification here rather than treating F2-A as globally redundant or incapable of helping elsewhere.

## 7. Explicit GO/NO-GO screens

Screens were fixed in the config before computing the preflight:

| Route | Decision | Observed quantity and rule |
| --- | --- | --- |
| G1: D0/F2-A gate | **NO-GO** | Epsilon headroom 0.003833 < 0.005. |
| G2: F0 accept/reject verifier | **GO** | Epsilon headroom 0.063819 >= 0.005, with advantage 0.059986 >= predeclared 0.005 material-advantage screen. |
| G3: three-way controller | **NO-GO** | Exact extra oracle value 0.000963 < 0.003. Unique-best prevalence is reported descriptively; no outcome-tuned alternative cutoff was added. |

**Recommended route: G2.** No significance test was invented for the headroom difference. GO recommends a bounded learnability study, not deployment, model fitting by default or further LM generation.

## 8. Action-specific target and class balance

For FIT supervision only:

```text
delta_action = saved_F0_F1 - saved_D0_F1
ACCEPT_THIS_F0 = 1 iff delta_action > 0.01
KEEP_D0 = 1 otherwise
```

Continuous delta is retained for research/possible future utility regression, but this first scaffold implements only the binary classifier ladder. No target means “some useful action exists.”

| F0 action state | Queries | Prevalence |
| --- | ---: | ---: |
| HELP: delta > +0.01 | 378 | 21.65% |
| NEUTRAL: abs(delta) <= 0.01 | 1003 | 57.45% |
| HURT: delta < −0.01 | 365 | 20.90% |

The binary ACCEPT target prevalence is **21.65%**. Accuracy from predicting KEEP everywhere would be uninformative; the primary evaluation is downstream selected-answer quality and the HELP-retention/HURT-rejection tradeoff.

## 9. Prepared verifier scaffold and leakage protections

The cheap scaffold materialized **32 inference-time features** and five outer paper folds, with no estimator construction/fitting:

- **V0 (11):** normalized y0/y1 equality, token edit distance/Jaccard, added/removed counts, answer lengths/difference/ratio, changed fraction and substring relation.
- **V1 lexical (5):** y0/E and y1/E overlap, overlap delta, added tokens present in E and removed tokens absent from E.
- **V2 length (2):** saved original/candidate generation lengths. These are lengths, not confidence measurements.
- **V3 (10):** top/second/third sentence cosine, mean/min selected cosine, next cosine/margin, evidence tokens, mean document-order source chunk rank and selected chunk count.
- **V4 (4):** query length, question mark and generic question-opening form indicators. No tuned domain rules or gold answer-type labels.

Optional q/y0/y1/E semantic similarities are not computed in this patch. Any future addition must use the frozen canonical MiniLM identity and a separately versioned feature protocol. Reliable logprob/entropy/EOS confidence is unavailable; no generation was rerun to obtain it. Missing saved rank/length values remain missing rather than observed zeros.

`online_features` accepts an exact whitelist of inference fields and rejects added outcomes/IDs. Matrix columns come only from an explicit feature registry; extra fields such as F1, HELP/HURT, gold/oracle data or paper IDs fail. Source IDs and paper IDs are audit/group metadata only. FIT labels live in a separate artifact. Imputation/scaling is confined to training-fold pipelines when training is later explicitly enabled.

### Lightweight model ladder, not run

- V-B0: standardized logistic regression, C=1, balanced class weights, fixed seed.
- V-B1: histogram gradient-boosted classifier with 100 iterations, at most 7 leaves, depth 3, min leaf size 20 and L2=1.

No LLM verifier, utility regressor, three-way model, query-to-K predictor or pre-gate cascade is trained or launched. Feature groups remain explicit for future controlled ablations; no model/feature proliferation was run during preflight.

### Paper OOF and threshold protocol

Future training is FIT-only with **five outer GroupKFold splits by paper_id**. Every query of a paper shares its held-out fold. **Four inner paper folds inside each outer training partition** produce inner OOF probabilities to select that outer model's threshold; outer-heldout scores never tune their threshold. Imputer/scaler/estimator fitting occurs on training rows only.

The threshold grid is predeclared: `{0,.05,.10,.15,.20,.30,.40,.50,.60,.70,.80,.90,.95,1,1.01}`. The last threshold permits KEEP-all. Maximize downstream selected-answer F1; within 0.003 prefer lower HURT, lower acceptance, lower candidate cost if it differs, then a simpler threshold closest to 0.5 and higher threshold as final deterministic tie-break. No default 0.5 assumption.

For later deployment, select a threshold from all outer OOF probabilities and fit the final estimator on FIT. This deployment selection is explicitly post-selection; it is not used to manufacture the honest nested-threshold OOF scores. Choosing between the two models on FIT is also post-selection and requires later independent confirmation.

Future controller outputs include actual selected-answer F1/EM, mean selected-action delta, HELP/HURT, acceptance, false acceptance, missed HELP, HURT rejection, HELP retention and neutral behavior, plus saved candidate costs and measured verifier prediction stage latency. False-accept rate is the non-HELP fraction among accepted proposals; harmful-accept precision is reported separately. Confusion counts distinguish ACCEPT/KEEP on actual F0-HELP/F0-HURT queries. No AUC-only success criterion is used.

## 10. Cost semantics: KEEP does not refund F0

The preflight oracle CSV reports conceptual **chosen-action** costs, excluding the work required to discover the oracle action. For example PRE-B epsilon reports approximately **1,176.54 prompt tokens and 1.2165 LM calls** because its retrospective output often chooses D0.

A real post-generation verifier cannot achieve that accounting: **F0 already exists before the verifier decides**. Every query therefore pays approximately **2,026.31 cumulative prompt tokens and two LM calls**, plus feature/verifier overhead, even when the final answer is D0. It is a quality/safety controller, not a compute-saving gate.

If both F0 and F2-A candidates were generated before three-way verification, shared initial-draft cost is counted once: approximately **3,205.76 cumulative prompt tokens and three LM calls**. These candidate-generation costs are saved separately in `post_generation_candidate_costs.csv`.

No full end-to-end latency is invented from cached first-pass, evidence or model stages. A future pre-generation gate/cascade is explicitly outside this patch.

## 11. Artifacts and code

Created:

- `src/week9/action_verification_g/{__init__,protocol,preflight,runner,features,controller}.py`.
- `configs/week9/w9g_action_verification.yaml`.
- `notebooks/week9/week9g_action_verification.ipynb`.
- `tests/week9/test_action_verification_g.py`.
- This implementation/results report.

Saved preflight: `results/week9/w9g_action_verification/v1/preflight/`:

- `protocol.json`, `preflight_manifest.json`;
- `action_values.csv`, `oracle_summary.csv`, `oracle_distribution.csv`, `oracle_choices.csv`;
- `three_action_unique_best.csv`, `three_action_query_categories.csv`;
- `action_targets_fit_only.csv`, `post_generation_candidate_costs.csv`;
- `headroom_analysis.json`, `go_no_go.json`, `w9g_preflight_decision.json`, `REPORT.md`.

Saved scaffold: `scaffold/inference_features.csv`, `feature_schema.json`, `fold_assignments.csv`, `manifest.json`. Metadata IDs are excluded from model columns. **No `controller_fit_oof` result, fitted model, OOF prediction or selected threshold exists.**

Checksums seal completed A–F/E3 inputs and legacy G files. The preflight's source identity covers only its own read-only runner/analysis/loader; future training separately pins feature/controller code, sklearn runtime, feature schema and config so adding the disabled scaffold does not reinterpret the completed preflight.

## 12. Tests and audit

**35 new tests passed.** They cover query/paper alignment and missing/duplicate rejection, exact/epsilon oracles and D0 preference, all seven tie classes, unique F2 contribution, GO/NO-GO screens, actual-F0 target and HELP/NEUTRAL/HURT thresholds, edit/evidence features, missing values, leakage/schema rejection, group folds and no paper overlap, post-generation KEEP cost, downstream threshold selection, no model/encoder/training during preflight, explicit training opt-in, legacy output-root protection, branch guard and default notebook behavior.

The broader Week-9 run passed **173 tests with 4 deselected**. The exclusions reflect saved notebook state outside this patch: three historical W9-PRE/E/E3 generic-default/prefix assertions, plus W9-F's default-disabled assertion after the user's completed full run left `RUN_F_FULL=True`. Those completed notebooks and tests were preserved. The new G notebook's own default path was tested: it invokes only the saved-output preflight, never training. Its 21-cell notebook schema validates.

The real full-FIT preflight and scaffold ran from saved artifacts, with post-run input hash checks. No real model fitting is done by the new tests. A reproducible independent saved-artifact audit and test logs are under `results/week9/patch_verification/w9g_action_verification/`.

## 13. Exact execution instructions

Open **`notebooks/week9/week9g_action_verification.ipynb`**, not the historical adaptive-K notebook. Defaults:

```python
RUN_PREFLIGHT = True
RUN_CONTROLLER_TRAINING = False
RUN_FULL = False
```

Run All safely recomputes the cheap saved-output preflight and displays its decision. Equivalent command:

```bash
./sara_env/bin/python -m src.week9.action_verification_g.runner --config configs/week9/w9g_action_verification.yaml --stage preflight --resume
```

**Do not run the following until controller training is deliberately approved after reviewing the preflight.** The exact future enabling flag is:

```python
RUN_CONTROLLER_TRAINING = True
```

Equivalent explicit future command:

```bash
./sara_env/bin/python -m src.week9.action_verification_g.runner --config configs/week9/w9g_action_verification.yaml --stage train --enable-training --resume
```

The runner rejects training without `--enable-training`, completed matching preflight and G2 GO. The notebook never trains just because GO was reported. Training would use saved candidates only and write a separate `controller_fit_oof` subtree; it would not generate answers or modify old G.

## 14. Remaining scientific and implementation limits

The high PRE-B oracle uses unavailable per-query outcome information. Inference features may fail to identify the useful changes, especially with answer noise and numerous score ties. F2-A's rare unique wins do not establish universal redundancy. Oracle distributions and a GO screen alone prove no learned verifier performance.

FIT was already used to select E3-k3 and F prompts. Future paper OOF tests the verifier conditionally on those globally frozen choices, not independent end-to-end development. Selected models/thresholds need later non-FIT confirmation under the appropriate evaluation protocol. No TEST claim or new held-out selection is made.

The disabled fitting path is scaffolded and structurally tested but has not been exercised with real fitted controllers; no quality/runtime result for either model is claimed. Optional semantic/confidence features remain unavailable. Exact source support, cosine and lexical overlap do not establish semantic entailment or causal evidence use. KEEP/ACCEPT verification cannot reclaim the cost of generating F0.

BRANCH = gagan

W9-G PREFLIGHT COMPLETED

CONTROLLER TRAINING WAS NOT STARTED

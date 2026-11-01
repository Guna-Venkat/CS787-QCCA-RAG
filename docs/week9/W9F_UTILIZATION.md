# W9-F — Utilization of frozen E3-k3 evidence

W9-F tests the generator interface while keeping evidence fixed. The existing W9-F notebook is patched to use the separate `src/week9/utilization_f` package and `configs/week9/w9f_utilization.yaml`. Historical source bundles, A–E3 artifacts and original E policy definitions are untouched. This implementation does not call W9-G training.

## Protocol and inputs

FIT only: 1,746 queries. FIXED-5 / SEMANTIC@20 / DraftSemantic requested coarse K=2 / query-only frozen MiniLM sentence ranking / sentence_k=3 / original-source-order presentation. **Load saved E3-k3 rows; never retrieve, rerank, segment, rescore, change k or rebuild evidence.** The generator remains trained SARA Mistral, FP16 efficient SDPA, canonical 4-FULL/6-HEAVY context and latent slots, greedy decoding, 128-token answer cap and the existing QA evaluator.

D0 remains the saved original draft and external SKIP action. It never receives a second answer-generation call. F0 is fresh regeneration after E3-k3 evidence exposure, not D0.

## Identity and F0 reuse

The read-only preparation step verifies the full existing A–E3 lineage, frozen config/source/model fingerprints, E3 protocol and full-run manifest, selected-k freeze, its artifact hashes, all saved row/call checksums and complete FIT query/k coverage. For each query it asserts:

- question/paper and original y0 match;
- source coarse chunk IDs, order and content hashes match;
- selected sentence IDs, exact source strings, text hashes and offsets match;
- presentation order and evidence hash match;
- D0 score equals the E3 saved initial score;
- reconstructed canonical F0 final prompt hash and initial token count match;
- checkpoint identity, tokenizer, generator interface/precision/attention, greedy decoding and cap match;
- the saved E3 LM call has that prompt, evidence IDs, decoding, model and answer.

Historical generation sources are pinned through E3's existing protocol, including the backend, context builder and revision interface. CPU reconstruction also validates the real canonical latent matrices and slot order. F0 aliases saved E3-k3 only after all checks pass. A failed equivalence check aborts; it never silently creates a different F0. A separate protocol would be needed if the generator/interface actually changed.

Every F1/F2 row retains the identical evidence text/hash, sentence IDs/text hashes and coarse source identity. Saved whole-source E3 inventories/caches are opened read-only where needed; F does not load the embedding encoder or compute embeddings. Hash seals are checked again after execution.

## Frozen prompt templates

Exactly **two** F2 variants are predeclared. Templates and hashes are saved in `prompts.json` and the protocol **before any smoke/full answer calls**. There is no iterative search after observing outcomes.

Each new revision message is:

```text
{instruction}

Initial answer: {initial_draft}

Additional evidence:
{exact_saved_E3_k3_evidence}

Question: {question}
```

This is inserted into the unchanged canonical SARA chat envelope/context. Only F1/F2 include y0. No gold/evaluation fields enter these messages.

### F1 — conservative preservation baseline

```text
The initial answer may already be correct. Review the additional source evidence. Preserve the original answer unless the evidence clearly requires a correction or addition. Make the smallest necessary change. Do not add unsupported information. Use only the supplied source evidence for any correction or addition. Output only the final concise answer, without reasoning or chain-of-thought.
```

### F2-A — direct evidence-anchored revision

```text
Answer the question using the supplied source evidence. Treat the source evidence as authoritative. The initial answer is only a starting point and may be incomplete or wrong. Preserve parts of the initial answer only when they remain supported by the evidence. Add missing information supported by the evidence, correct conflicting claims, and remove unsupported material when necessary. Do not invent facts. Output only the final concise answer, without reasoning or chain-of-thought.
```

### F2-B — explicit missing/conflicting/supported content

```text
Answer the question using the supplied source evidence as the authoritative source. Use the initial answer as a starting point, not as an authority; it may be incomplete or wrong. Consider information missing from the draft, claims contradicted by the evidence, and supported draft content worth preserving. Add supported missing information, correct contradictions, and remove unsupported material when necessary. Do not invent facts. Output only the final concise answer; do not output these categories, reasoning, or chain-of-thought.
```

No F2-C is added. The exact F0 template remains the existing `revision_messages(..., False)` behavior rather than a rewritten baseline.

## Selection and freeze rules

Both freezes are FIT-only and require complete full coverage. Smoke cannot select or freeze anything.

1. Select F2 among **F2-A/F2-B**.
2. Select one intervention among **F0/F1/frozen F2**. D0 is excluded and remains SKIP.

For each selection: find maximum FIT F1; retain choices within **0.003**. Preserve the existing project's **cost-first** convention: lower cumulative query-time prompt tokens, then materially lower HURT tier, lower generated tokens, lower recorded answer-stage latency, shorter template, stable policy name. A candidate at least **0.01 absolute HURT** above the minimum eligible HURT is placed in the higher-risk tier. This threshold and ordering are preregistered, not tuned on results. Cost precedes HURT to retain the existing convention explicitly allowed in the request.

Save exact prompts/hashes, every candidate's FIT metrics, protocol/config/generator/source identities, evidence policy and the selection rule. The intervention artifact distinguishes high-level mode F2 from its actual F2-A/B prompt. It never hard-codes F1/G1 as the future intervention.

## Metrics and mechanism analysis

Quality summary includes D0, F0, F1, both F2 candidates and (full FIT only) a clearly identified selected-F2 alias: n, F1, EM, delta vs D0/F0, HELP, HURT and strict answer change vs D0. HELP/HURT use >0.01 / <−0.01 F1 changes. Also report normalized answer change and F1 differences vs F0, exact/normalized draft preservation, normalized intervention, descriptive HELP/(HELP+HURT), and mean delta among HELP/HURT queries.

For each policy, stratify queries by **F0 HELP / HURT / approximately unchanged** relative to D0. Report group counts, mean delta vs D0/F0, preservation, HELP/HURT and:

- Partial rescue: among F0-HURT queries, revised F1 > F0 F1 + 0.01.
- Full recovery: among F0-HURT queries, revised F1 ≥ D0 F1 − 0.01.
- HURT rescue: among F0-HURT queries, revision is no longer HURT vs D0.
- HELP retention: among F0-HELP queries, revision remains HELP vs D0.
- Lost gain: among F0-HELP queries, revision is no longer HELP vs D0.
- Diminished gain: among F0-HELP queries, revised F1 < F0 F1 − 0.01.

Report overall HURT and HELP rate reductions vs F0. Empty conditional denominators are null, not zero. Normalized equality uses lowercase, punctuation removal and whitespace normalization; strict equality is separately available. Lexical added/removed token counts and answer/evidence overlap are descriptive, not causal or correctness evidence. No new judge, answer-type labels or outcome-tuned difficulty strata are introduced.

## Costs and failure behavior

Every policy retains conceptual initial-draft prompt/generated tokens. F0 reuses its saved E3 costs, and F1/F2 expose y0 explicitly in longer prompts. Cumulative prompt/generated tokens, LM call counts, evidence tokens, retrieval/selector latency, inherited embedding/index/ranking costs and measured answer-stage latency are separate fields. No cached sentence embedding is represented as conceptually free or recomputed for F.

`physical_new_answer_calls_this_invocation` distinguishes newly executed calls from cached or aliased outputs. F0 causes no new answer calls. Per-row `new_answer_calls` records calls made when that result row was created, not the current resumed invocation. Saved LM-call stage latency survives resume; it is not newly measured on cache hits. First-pass, retrieval/index work are cached; measured end-to-end latency remains null.

New F1/F2 empty or recognizably invalid answer wrappers/reasoning envelopes retain exact D0 with an explicit answer-level failure reason. The attempted call/tokens remain charged. Raw generated text is saved for audit. No extraction-policy failure is invented, no policy winner is substituted and answers are not silently rewritten to repair wrappers. Ordinary output uses the unchanged answer evaluator. F0 retains its historical E3 parser behavior rather than retrospectively altering the baseline.

## Statistical comparisons

Full FIT only: paired **paper-cluster bootstrap, 10,000 resamples, seed 42**. Report F0−D0, F1−D0, F1−F0; each F2 candidate against D0/F0/F1; and selected-F2 against D0/F0/F1. Selected-F2 comparisons are explicitly post-selection. Intervals are unadjusted; positive point estimates with intervals crossing zero establish no superiority. All variants are logged; no held-out split or TEST is inspected to tune prompts.

## W9-G handoff — no training

Only complete FIT analysis creates `w9f_controller_handoff.json`: frozen utilization mode/prompt/config/generator identity, per-query IDs/papers, evidence hashes and sentence IDs, D0/intervention scores, delta/HELP/HURT and costs. Outcome labels are explicitly **FIT research only, forbidden as inference features**.

W9-G must wait because E3's oracle used F0-like regeneration, while F1/F2 can change query-level gains, harms and skip/intervene utility. Any future controller must consume the verified W9-F intervention freeze. W9-F does not train a gate, dynamic k, STOP/ADD policy or action selector, and does not modify the legacy G implementation.

## Artifacts

Under `results/week9/w9f_utilization/v1/`:

- `protocol.json`, `config_snapshot.json`, `prompts.json`;
- `f0_alias_verification.json`, `source_evidence_identities.json`;
- `smoke/results.jsonl`, `smoke/lm_calls.jsonl`, identities, summary/cost/transition/rescue/lexical diagnostics, manifest/report;
- after manual full execution: corresponding `fit/` artifacts plus `bootstrap.csv`;
- after complete FIT only: `frozen_w9f_f2.json`, `frozen_w9f_intervention.json`, `w9f_controller_handoff.json`.

Full result/call JSONL files are ignored by Git; compact metadata, summaries, smoke and freezes remain reviewable. Independently resumable LM-call/result caches use separate F-only protocol/source/prompt/input identities. This keeps historical global A–E3 source bundles intact.

## Manual execution

Open **`notebooks/week9/week9f_revision.ipynb`** from disk. Default flags are safe for Run All:

```python
RUN_F_FULL = False
RUN_F_SMOKE = False
```

Inspect saved smoke, then manually enable **only**:

```python
RUN_F_FULL = True
RUN_F_SMOKE = False
```

Equivalent command from the repository root:

```bash
./sara_env/bin/python -m src.week9.utilization_f.runner --config configs/week9/w9f_utilization.yaml --stage full --resume
```

This is the new controlled E3-k3 experiment. The historical `run_w9f_revision.py`/`w9f.yaml` pathway remains preserved for provenance and is not the execution route for this experiment.

CPU-only preparation (also performed automatically by smoke/full):

```bash
./sara_env/bin/python -m src.week9.utilization_f.runner --config configs/week9/w9f_utilization.yaml --stage prepare --resume
```

Tiny smoke uses `--stage smoke --resume`. Analysis from completed saved FIT uses `--stage analyze --resume` and loads no generator. Full costs up to **5,238 new revision conditions** (three per query); D0 and F0 are reused. Do not change prompts/code/config/model identities mid-run. A mismatched cache aborts rather than mixing conditions.

## Validation and limits

See `results/week9/patch_verification/w9f_utilization/` for the actual test/smoke report and inspection. No full W9-F is started by implementation or tests. Smoke quality cannot establish F1 preservation, F2 correction ability or superiority; the fixed prompts may fail. FIT has already been used heavily, so later non-FIT confirmation is essential. Source-grounded sentences are not guaranteed relevant/sufficient/causal or semantically faithful; source authority is a prompt instruction, not a claim that every selected sentence answers the question.

### Actual verification performed

- CPU-only preparation proved **1,746/1,746 F0 aliases equivalent**; no generator loaded for that step.
- **22 new tests passed**, including exact-source/identity tampering, F0 alias model/prompt/call mismatch rejection, D0 loading, distinct draft-aware prompt content, gold rejection, frozen prompt hashes and two-variant limit, HELP/HURT, preservation, rescue/full recovery, lost/diminished gain, transition groups, paper bootstrap pairing/reproducibility, cost-tolerant selection, full-only immutable freezes/handoff, empty/wrapper fallback with charged calls, no extra selection/controller invocation, unchanged source snapshots, deterministic smoke IDs and disabled notebook execution.
- **139 Week-9 tests passed, 3 deselected**. The three legacy checks were individually observed to conflict with the pre-existing saved notebooks: `test_notebooks_have_safe_defaults_and_no_test_dropdown` expects generic flags in W9-PRE; `test_e_notebook_disabled_path_executes_without_generation` conflicts with the saved E3 full-run flag; `test_old_notebook_prefix_preserved_and_all_flags_disabled` compares an older serialized E-prefix snapshot with subsequently executed outputs. W9-E/E3 and those tests were preserved rather than edited to force a pass. W9-F's own default-disabled execution test passes after smoke artifacts exist.
- The patched **30-cell notebook** passed notebook schema validation. `git diff --check` passed.
- A real deterministic FIT smoke produced **10 rows**: D0/F0/F1/F2-A/F2-B for **two queries**, **six newly executed revision calls**, **zero extraction calls**, identical frozen evidence across all intervention modes and **zero answer fallbacks or detected wrapper/reasoning outputs**. Peak allocated GPU memory was **13.84 GiB**.
- Existing A–E3 input hashes and all saved E3 files were checked unchanged after execution. A separate read-only smoke audit independently rechecked those hashes and row/call checksums.
- No `fit/` directory, F2 winner freeze, intervention freeze or controller handoff exists under the new F result root. Those are created only by a later manually enabled complete FIT run.

### Smoke inspection

| FIT question | D0 | F0 (reused) | F1 | F2-A | F2-B |
| --- | --- | --- | --- | --- | --- |
| 218: “Did the authors use crowdsourcing platforms?” | Yes | Yes | Yes | Yes | Yes |
| 1876: “Do the authors give any examples of major events which draw the public's attention and the impact they have on stock price?” | No | Yes | No | No | Yes |

All answers were concise answer-only outputs. The first query illustrates preservation with compatible evidence; the second shows the planned preservation/lost-gain distinction. F0/F2-B received higher QA scores on the second example, but the three selected sentences alone do not explicitly establish the entire answer; neither score nor a changed answer proves that the model grounded its decision in this evidence. No prompt is selected or revised from this sample.

Exact question text, drafts, source chunk/sentence IDs, evidence strings/hashes, answers and prompt token counts are saved in `results/week9/patch_verification/w9f_utilization/verification.json`. `verify_smoke.py` reproduces the read-only audit. Test logs and the smoke log are beside it. No extra examples were generated to obtain a favorable correction demonstration.

### Files created or changed

- New implementation: `src/week9/utilization_f/{__init__,prompts,protocol,analysis,runner}.py`.
- New config: `configs/week9/w9f_utilization.yaml`.
- Patched existing notebook: `notebooks/week9/week9f_revision.ipynb`.
- New tests: `tests/week9/test_utilization_f.py`.
- Documentation: this report and a W9-F entry in `docs/week9/WEEK9_RUNBOOK.md`.
- `.gitignore`: only new W9-F full result/call cache patterns.
- New separate protocol/alias/smoke artifacts under `results/week9/w9f_utilization/v1/` and verification artifacts under `results/week9/patch_verification/w9f_utilization/`.

Historical top-level Week-9 sources, scripts, old F configuration and all completed A–E3 scientific artifacts are untouched. No commit, push, full W9-F execution or W9-G training was performed.

BRANCH = gagan

FULL W9-F RUN WAS NOT STARTED

# W9-E v1 — compact, mechanically grounded evidence

Implementation is complete; **the structured-policy preflight is not a clean quality pass**. Do not treat this as an unattended full-run recommendation. The default gate requires review and explicit acknowledgment of the saved quality warnings. No full W9-E experiment was started.

## Scope and frozen lineage

Work is confined to branch `gagan`. Existing A/B/C/D source files, results, frozen artifacts and fingerprints are unchanged. The historical global source hash covers `src/week9/*.py` and `scripts/week9/*.py`; the new implementation is deliberately in the nested `src/week9/grounded_e/` package, with its **own explicit file hashes** included in every cache identity. The existing E notebook now invokes that package. The old E CLI is preserved for provenance and must not be used for this grounded protocol.

The fixed diagnostic inputs remain FIXED-5 / SEMANTIC@20, DraftSemantic, requested K=2 and the saved initial draft. Primary B stays K=0. Every D query must have exactly D0/D1/D2/D3; positive-arm IDs, ordering, paper identities and source text must agree. No retrieval or selector is rerun. D was complete but had no separate frozen D configuration; an **E-local immutable D-input seal** pins its existing row file, identity, completion manifest and prerequisite freezes without rewriting any of them. Every E row stores corresponding D row hashes and chunk text hashes.

## Files created or modified

- Modified: `.gitignore` (only the new E full-run JSONL caches; preflight/summaries stay reviewable).
- Modified: `notebooks/week9/week9e_extraction_faithfulness.ipynb` (the existing notebook, all fourteen requested sections).
- Created: `src/week9/grounded_e/__init__.py`, `prompts.py`, `evidence.py`, `protocol.py`, `runner.py`, `analysis.py`.
- Created: `configs/week9/w9e_grounded.yaml` and `configs/week9/w9e_grounded_64.yaml`.
- Created: `tests/week9/test_grounded_e.py`.
- Documentation: this report, E-only additions to `docs/week9/WEEK9_RUNBOOK.md` and the E section of `docs/week9/WEEK9_LIVE_TRACKER.md`.
- Verification: `results/week9/patch_verification/w9e_grounded/{preflight.log,verify.py,verification.json}`.
- New E-only protocol, cache and preflight outputs: `results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/`.

No existing frozen artifact or A-D result was overwritten. No model, SFR/projector, selector, retrieval or F revision-mode change was introduced.

## Policies, exact prompts and budgets

| Policy | Mechanism | Draft used for extraction? |
|---|---|---|
| E0 | Deterministic query-only BM25-ranked source sentences, greedily packed exact word-boundary spans | No |
| E1-Q | Joint strict-JSON support span plus a short copied fact | No |
| E1-QD | Same schema, with source-authoritative draft correction/completion instructions | Yes |
| E2 | Same grounding with minimal type/relevance blueprint labels | Yes |

Exact templates are in `src/week9/grounded_e/prompts.py`; the pinned template snapshot is `.../budget_128/protocol/prompt_templates.json`. `COMMON` plus `STRUCTURED` implements E1-Q; E1-QD additionally uses `DRAFT`. E2 uses `COMMON`, `BLUEPRINT` and `DRAFT`. The common budget placeholder resolves to the configured final evidence budget. Actual per-query extraction prompt hashes, source chunk hashes, policy identities and decoding settings are saved for reproducibility. No prompt repair or outcome-driven prompt iteration was performed.

E2 adapts the existing local SANE blueprint concept (support plus compact organization), without copying its verbose free-form table interface. SANE originals remain unchanged. The model cannot produce free factual paraphrases here: a fact/claim must be a contiguous phrase copied from its support span, with only whitespace normalization in the final fact. This conservative variant is intentionally narrower than arbitrary normalized facts. The prompt requests at most two items, support at most 24 tokens, fact/claim at most 12 tokens, no rationale or chain-of-thought.

The final **entire serialized evidence representation** (source labels, quotes, JSON keys, punctuation included) must fit 128 tokens. The alternate 64-token configuration uses a separate output directory and requires its own preflight. E0 trims only exact source prefixes at word boundaries; structured items that exceed the item or whole-object budget are dropped, not rewritten. Intermediate structured extraction has a strict 256-token generation cap; no increase to D's 1,024 cap is hidden. The answer cap stays 128.

## Grounding and fail-closed semantics

Validation requires an exact substring in the specified selected chunk; no fuzzy matching or semantic similarity is allowed. This implementation chooses strict matching rather than even optional whitespace/Unicode span repair. Accepted spans record query/paper/chunk IDs, exact canonical text, character offsets and token counts. Structured claims are also checked as copied contiguous subspans. Labels are constrained to enumerated short values.

Malformed/empty/unsupported/entirely rejected/over-budget extraction produces no evidence and no downstream answer call: the final answer is exactly y0. Extractor cost remains charged. There is no raw-chunk or synthetic fallback. Partial rejection can retain other valid budget-fitting items. Accepted quote support is mechanical evidence provenance, **not semantic entailment, relevance, sufficiency or assurance that omitted context cannot change meaning**.

## Preflight results and remaining risks

The deterministic seed/hash FIT sample is query IDs **2038, 1583, 567**. No DEV or INTERNAL-VAL data was used. Three queries x four policies gave 12 saved rows, 13 new model calls, maximum accepted evidence length **110 tokens**, peak GPU allocation **14.21 GiB**. All accepted spans were reconstructed from source offsets. All eight fail-closed rows kept y0 with **no answer call**. A-D source/input hashes were rechecked after the smoke and remain unchanged. GPU model memory was released after the process exited.

| Policy | Usable evidence | Mean evidence tokens (including failures) | Median tokens | Malformed | Fail closed | Generation-cap hits |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 3/3 | 96.67 | 100 | 0/3 | 0/3 | 0/3 |
| E1-Q | 1/3 | 15 | 0 | 0/3 | 2/3 | 0/3 |
| E1-QD | 0/3 | 0 | 0 | 2/3 | 3/3 | 0/3 |
| E2 | 0/3 | 0 | 0 | 2/3 | 3/3 | 1/3 |

E0 compression averaged .0451 relative to selected raw chunk tokens. Proposed exact support was 1.0 for E0, .3333 for E1-Q and 0 for E1-QD/E2; **all accepted items had exact support**. Raw-empty output rate was zero, although rejected outputs leave no usable evidence. E1-Q's single accepted object was merely a copied word, illustrating that exact support does not establish utility. Detailed p90 lengths, support/fact tokens, acceptance/rejection counts, source-verification examples and raw outputs are in the preflight files.

The current SARA extractor still struggles with strict compact schemas. These are observed model/interface failures, not a successful structured acquisition result. A three-query preflight is too small for statistical claims and its downstream F1 was not used to tune/select methods. The full-run gate remains blocked by default because the warning rates are high. Reviewing/acknowledging them is a deliberate experimental decision, not an implicit claim that the structured conditions work. No additional prompt variant was tried to chase an F1 gain.

## Analysis and cost accounting

Full analysis imports D0/D1/D2/D3 read-only alongside E0/E1-Q/E1-QD/E2. It reports F1, exact match, mean delta F1, HELP/HURT and answer-change rates relative to y0, plus explicitly descriptive HELP/(HELP+HURT). Evidence tables report proposed and accepted exact support separately, raw-empty/invalid/fail-closed/cap rates, mean/median/p90 final lengths, compression, zero-support rates and accepted/rejected facts/items, fact tokens and support tokens. Zero-output failures must not be mistaken for effective compression.

Eight required comparisons use the existing deterministic paper-cluster bootstrap, 10,000 resamples, seed 42. Only complete full-FIT coverage permits intervals: E0-D0, E0-D1, E1-Q-D0, E1-QD-D0, E1-QD-E1-Q, E1-QD-D3, E2-D0, E2-E1-QD. Positive point estimates alone establish no superiority. Negative results with low usable-evidence rates do not cleanly falsify all possible compact grounded methods.

The same canonical SARA FP16 efficient-SDPA backend/context/latent embeddings and D regeneration behavior are used. Extraction, selection/retrieval, answer and cached original-draft tokens/calls are accounted for separately. Final answer prompt tokens and cumulative prompt/generated tokens remain distinct. Extraction calls are fully charged even if the answer fails closed. Candidate text in memory is not automatically generator input. Initial generation latency is unknown; inherited retrieval/selector and cached model component timings are labelled; measured end-to-end latency stays null.

## Tests and verification

- **103 passed, 1 deselected** in the W9 suite. The existing global notebook-default test was excluded because the user enabled B/C/D execution flags; the new dedicated E notebook test executes every cell with full/smoke flags disabled and performs no generations.
- **23 new E tests** cover exact offsets, fabricated spans/IDs, malformed E1/E2 schemas, invalid labels/oversized facts, fail-closed answer-call suppression and charged extraction, 64/128 whole-representation budgets, selected chunk identity, cache policy/prompt/budget/source separation, query-only draft exclusion, gate tampering/staleness, read-only D loading, JSON-safe identities and unchanged historical source bundles.
- Existing bootstrap helper was reused without modification; a tiny-fixture integration test checks all eight required intervals and suppresses unsupported superiority claims. Independent real-preflight verification re-tokenized all accepted evidence with the actual tokenizer and checked row/cache checksums and unchanged A-D files.
- No full E run, full A-D rerun, E winner freeze, F/G run, DEV/INTERNAL-VAL tuning, TEST access or projector training occurred.

## Exact manual instructions

Reload `notebooks/week9/week9e_extraction_faithfulness.ipynb` from disk. Defaults are RUN_FULL=False and RUN_SMOKE=False; the saved preflight and warnings are displayed without generations. To revalidate the identical smoke, enable RUN_SMOKE=True with RUN_FULL=False; compatible saved rows/calls resume. To deliberately run the full experiment **after reviewing the warnings**, use RUN_FULL=True, RUN_SMOKE=False, ACK_PREFLIGHT_WARNINGS=True. Do not alter protocol/config/source files mid-run; compatible `--resume` reuses completed E rows and calls. A new budget/config requires its own preflight.

```bash
# Already executed: deterministic small FIT preflight only
./sara_env/bin/python -m src.week9.grounded_e.runner --config configs/week9/w9e_grounded.yaml --stage preflight --resume

# Manual full run only after reviewing/acknowledging the warnings — NOT EXECUTED
./sara_env/bin/python -m src.week9.grounded_e.runner --config configs/week9/w9e_grounded.yaml --stage full --resume --ack-preflight-warnings

# Rebuild analysis from a complete saved E run without generation
./sara_env/bin/python -m src.week9.grounded_e.runner --config configs/week9/w9e_grounded.yaml --stage analyze --resume
```

**FULL W9-E RUN WAS NOT STARTED**

**BRANCH = gagan**

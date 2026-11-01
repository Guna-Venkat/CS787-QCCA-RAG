# W9-E3 — Semantic Sentence Suppression

E3-SentTopK is implemented as a **new experiment**, appended to the existing W9-E notebook. The full E3 run has not been started. Existing E0/E1-Q/E1-QD/E2 definitions, results and frozen input fingerprints remain unchanged. The older notebook cells and saved outputs are preserved; only the existing RUN_FULL/RUN_SMOKE defaults were restored to False for safe execution.

## Files and isolation

New package: `src/week9/senttopk_e3/{__init__,sentences,embeddings,protocol,analysis,runner}.py`.

New config: `configs/week9/w9e3_senttopk.yaml`.

New tests: `tests/week9/test_senttopk_e3.py`.

Modified notebook: `notebooks/week9/week9e_extraction_faithfulness.ipynb`, with a clearly separated appended E3 section and fourteen subsections. No earlier section was renamed or old policy reinterpreted.

Documentation: this report plus an E3-only runbook addition. `.gitignore` adds only new E3 large-cache patterns. Verification and notebook-prefix audit are under `results/week9/patch_verification/w9e3_senttopk/`.

All new results live under `results/week9/w9e3_senttopk/v1/`. New package sources have explicit independent hashes; neither `src/week9/*.py` nor the existing grounded-E package is changed. Thus A-D's historical global source bundle and E's dedicated source bundle remain intact. Existing D/E results are read and checksum-verified, never regenerated, renamed, frozen anew or overwritten.

## Exact fixed protocol

- Data: FIT only; seed 42 with the existing Python/NumPy seed convention and greedy SARA decoding.
- Frozen retrieval: FIXED-5 / SEMANTIC@20.
- Coarse selection: the **same saved** DraftSemantic requested K=2 chunks, preserving original selected IDs and exact content hashes. A query with fewer than two actual chunks uses its original saved selection; no missing candidates are fabricated.
- Primary policy remains K=0. The E3 branch is a forced-intervention diagnostic.
- Scoring: **question only**, never draft, gold, F1, HELP/HURT or oracle information.
- Sentence counts: **{1,2,3,4,6,8}**, each evaluated and reported separately as E3-k1/k2/k3/k4/k6/k8.
- Evidence: exact selected canonical sentences joined by blank lines, without labels, paraphrases, claims, blueprint or extraction LM.
- Presentation: restore selected sentences to original document order, regardless of score order.
- Generator: unchanged trained SARA Mistral FP16 efficient SDPA, canonical 4-FULL/6-HEAVY context/latent embeddings, original D revision=False prompt semantics and greedy 128-token output cap. Same answer parser/evaluator. There is no F-mode experiment or new model.
- No new evidence token cap is imposed: this is a fixed **sentence-count** surface, not a fixed-token comparison. Complete sentences are never silently clipped. The normal generator context-capacity check still fails loudly on overflow.

A source mismatch is fatal. If selected chunks yield no valid sentences, final prediction is exactly D0/y0 with `fail_closed=true`, `failure_reason=no_valid_sentences`, and no answer call. Empty model answers, if any, are evaluated with the unchanged D parser rather than silently replaced with a new policy.

## Sentence splitter

The existing repository splitter breaks at abbreviations indiscriminately. E3 therefore uses the independent stable rule `academic-offset-rules-v1`: punctuation and paragraph boundaries, with a fixed academic-abbreviation list and rules for initials/acronyms. Tests cover Dr., et al., Fig., Eq., e.g., decimals and author initials. No external language model, mutable pretrained segmentation resource or LLM is used.

Leading/trailing whitespace is omitted only by moving offsets; all retained text is the exact canonical source substring, including internal whitespace. Every sentence stores question/paper/chunk/sentence IDs, chunk hash, chunk-relative and document-relative character offsets, token count, local order, original document-order chunk rank and original selected-coarse-chunk rank. Sentence IDs hash source identity, offsets and splitter version.

Document order is recovered from frozen chunk `char_start`/`char_end`, not DraftSemantic's relevance order. Coarse fixed chunk edges can split an original document sentence; resulting edge fragments are retained as exact source regions and explicitly documented. Abbreviation ambiguity and tables can still cause imperfect linguistic segmentation; this is a deterministic source-preserving rule, not a claim of perfect sentence semantics.

## Encoder, ranking and presentation

The canonical encoder is `sentence-transformers/all-MiniLM-L6-v2`, frozen local snapshot:

`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`

All snapshot weights/config/tokenizer files must match upstream frozen model hashes. The unchanged `LocalEncoder` uses nonoverlapping encoder-token windows, mean pooling and L2 normalization; long sentences are not silently truncated. Query and sentence vectors inhabit the same space. Scores are clipped normalized dot products (cosine).

Tie-break order is descending cosine, lower original document-order chunk rank, lower local sentence order, stable sentence ID. Top-k selection pools **both** saved chunks. If fewer than k sentences exist, actual k is recorded and all available IDs are used exactly once.

After selection, presentation sorts by document character offset with deterministic source tie-breaks. Both `selected_rank` and `presentation_rank`, plus separate ranking-order/presentation-order ID arrays, are saved. No MMR, deduplication-driven selection change or diversity reranking is introduced.

## Cache strategy and cost distinctions

Indexed, losslessly compressed SQLite caches hold sentence and query vectors separately. Keys include exact text hashes, embedding model/revision/file hashes and normalization settings; cache identity also includes E3 sources/config. Inventories and per-query full sentence scores are separately saved with checksums. Safe single-writer caches share only compatible E3 embeddings across smoke/full stages. Incomplete embedding batches may be recomputed on interruption; completed answer calls/results remain independently resumable.

Costs explicitly distinguish:

- unique sentence-embedding preprocessing/index computation;
- query embedding conceptual cost (retained even when reused);
- current-run cache lookup and computation;
- sentence ranking origin/current time;
- inherited frozen coarse retrieval/selector time;
- final answer generation time;
- final prompt and cumulative query-time prompt/generated tokens, including the saved initial draft;
- conceptual per-fixed-policy LM calls versus physical calls in the shared experiment cache.

Cached embeddings are never portrayed as free conceptual preprocessing. Candidate text in memory is not charged as LM input unless actually included in a prompt. End-to-end query latency remains null because the initial draft is cached. Per-policy costs describe executing that action; physical embedding-cache/index totals are reported separately rather than multiplying shared preprocessing by six policies.

## Analysis, freeze and oracle diagnostics

Full analysis imports D0/D1/D2/D3/E0/E1-Q/E1-QD/E2 read-only alongside all fixed E3 conditions. It reports n, F1, exact match, delta/HELP/HURT/answer change versus D0; HELP/(HELP+HURT) is descriptive only. Evidence metrics include actual/requested k, mean/median/p90 tokens and compression, mean/min selected cosine, next-unselected cosine/margin and exact source validation.

Diversity diagnostics include same-chunk pair fraction/all-same-chunk indicator, adjacent-sentence **pair** fraction, pairwise cosine, exact duplicates and cosine>=.95 near-duplicate pair rate. Pair statistics are undefined for fewer than two selected sentences, rather than fabricated as zero. No gold-sentence overlap metric is reported: a validated exact mapping to this new sentence inventory was not prepared, so coarse overlap is not mislabeled as sentence recall. No gold enters ranking.

The required 10,000-resample paired **paper-cluster** bootstrap (seed 42) compares each fixed k to D0/D1 and the FIT-selected fixed policy to D0/D1/E2. These are FIT diagnostics; best-policy intervals are post-selection descriptive evidence, and individual fixed-k intervals are not multiplicity-corrected. Positive point estimates alone establish no superiority.

The preregistered global fixed-k rule maximizes FIT F1, then among candidates within .003 selects lower cumulative prompt tokens, lower final prompt tokens, smaller k. Complete FIT coverage alone writes the separate `frozen_e3_senttopk.json`, including all candidates, full protocol/model/source/config identities and hashes of result/analysis artifacts. No old freeze is replaced.

Separate retrospective oracles:

1. Intervention-only: {1,2,3,4,6,8}.
2. Full action: {0,1,2,3,4,6,8}, with k=0 imported from D0.

Exact ties choose smallest k. Epsilon=.01 chooses smallest qualifying k. Each reports F1, mean/median k, k distributions and chosen-action evidence/final/cumulative prompt/generated token costs and LM calls. Oracles are **not deployable**; these costs exclude the search needed to discover oracle actions.

An additional local-window diagnostic uses only already evaluated k satisfying |k-k0|<=3. It does not generate or invent k=5/7 and does not alter the primary six-value grid.

## Dynamic-k GO/NO-GO

Only a retrospective recommendation is implemented. No E3B controller is built, trained or run automatically. The comparison benchmark is the **highest-point-F1 fixed E3 policy**, avoiding artificially inflated headroom from a possibly lower-F1 cost-tolerant choice.

GO requires at least one preregistered screen:

1. Full-action epsilon oracle improves F1 by >=.005.
2. It remains within .003 F1 while reducing at least one explicitly reported evidence/final/cumulative prompt cost by >=15%.
3. Intervention-only epsilon choices have at least two k values each with >=15% mass, fixed-k HELP or HURT varies by >=.05, and full-action oracle gain is >=.003.

Otherwise NO-GO. Evidence-token savings alone are not described as total-query savings. Full-action headroom includes avoiding intervention; inspect k=0 mass and degeneracy before treating a numeric GO screen as evidence that sentence-count control is useful. A GO result authorizes no automatic training and proves nothing about an inference-time controller. No held-out data is inspected to select k.

## Tiny smoke and tests

Actual smoke: two deterministic hash/seed-selected FIT queries, six k values each, **12 rows and 12 answer-LM calls, zero extraction-LM calls**. No k was selected or frozen; no oracle/dynamic decision was produced from the smoke.

- Query 594: “What features are proposed?” — 75 candidate sentences from the same two D/E chunks.
- Query 1204: “What are the comparable alternative architectures?” — 76 candidate sentences.
- Selected evidence ranged from **18 to 196 tokens** across k/query combinations.
- Peak allocated GPU memory: **13.88 GiB**.
- All source substrings and offsets validated. Cosines/ranking were independently recomputed from the persisted vectors; every selected prefix and restored presentation order matched.
- All prior D/E artifact hashes and historical source bundles were verified unchanged after execution.
- Full E3 output and fixed-k freeze do not exist.

All candidate sentences and scores, question text, k-specific selected/presentation ranks, exact evidence and generated answers are available in the smoke/cache artifacts and displayed by the appended notebook section. Smoke quality is not used to choose k or tune the method.

**119 passed, 1 deselected** in the W9 suite. The deselected existing all-notebooks defaults check conflicts with user-enabled B/C/D flags; dedicated E3 safe execution passed before and after real smoke artifacts existed. Sixteen new tests cover deterministic academic segmentation, offsets/source forgery, cosine/tie rules, top-k/actual-k, restored order, no gold in scoring, cache identities/nonzero conceptual cost, frozen encoder revision, D/E chunk matching, read-only old E artifacts, D0/intervention oracles, epsilon/smallest-k ties, complete bootstrap/freeze/oracle integration, dynamic NO-GO, no-sentence fallback and disabled notebook execution. Existing bootstrap helper and historical source files were not modified.

## Artifacts

Under `results/week9/w9e3_senttopk/v1/`:

- `protocol.json`, `source_config_prompt_model_hashes.json`;
- `sentence_cache/inventories.jsonl`, `scores.jsonl`, sentence/query SQLite embedding caches and identity metadata;
- `smoke/results.jsonl`, `lm_calls.jsonl`, `summary.csv`, `embedding_costs.json`, `manifest.json`, `REPORT.md`;
- after manual full execution only: `fit/results.jsonl`, `lm_calls.jsonl`, `summary.csv`, `bootstrap.csv/json`, exact/epsilon oracle choice/cost/distribution/summary files, `dynamic_go_no_go.json`, embedding cost and completion manifest/report;
- after complete full FIT only: `frozen_e3_senttopk.json`.

Large caches/full generation JSONL files stay local and ignored by Git; protocol/identity metadata, compact smoke results, summaries and verification remain reviewable. Independent audit: `results/week9/patch_verification/w9e3_senttopk/verification.json`; reproducible `verify_smoke.py` and notebook-prefix audit are alongside it.

## Manual execution

Reload the existing W9-E notebook from disk. Keep old `RUN_FULL=False` and `RUN_SMOKE=False` so E0–E2 are not re-executed. In the appended **W9-E3 — Semantic Sentence Suppression** section, default `RUN_E3_FULL=False`, `RUN_E3_SMOKE=False` safely displays saved smoke results without generations.

To deliberately run the full surface:

```python
RUN_E3_FULL = True
RUN_E3_SMOKE = False
```

Or use:

```bash
./sara_env/bin/python -m src.week9.senttopk_e3.runner --config configs/week9/w9e3_senttopk.yaml --stage full --resume
```

To rerun only the same compatible tiny smoke, use `--stage smoke --resume`. To rebuild analysis from a complete saved full surface without generation, use `--stage analyze --resume`. Do not edit protocol/model/config/source files mid-run: incompatible identities fail loudly; use a separately versioned experiment instead.

There are up to **10,476 answer conditions** in the full surface. Removing extraction-LM work does not make six answers/query free; the two-query sample does not provide a reliable full-runtime estimate. The primary method may still fail to improve quality. Source-grounded does not mean relevant, sufficient or answer-correct; high cosine does not establish causality or factual support. No general-RAG or dynamic-controller success claim is made.

BRANCH = gagan

FULL W9-E3 RUN WAS NOT STARTED

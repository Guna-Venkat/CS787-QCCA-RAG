> **Final-freeze update:** `evidence-lab-demo-1.0.0` selectively versions source, tests, tokenizer assets, documentation and a 13.93 MiB immutable archive of the unchanged curated bundle. Controlled retriever comparison is the presentation default; native variants and historical lexical QBGF are researcher-only. See [FINAL_FREEZE_STATUS.md](FINAL_FREEZE_STATUS.md) and [archive restore instructions](releases/README.md). Earlier implementation chronology below describes the development state at the time.

# Demo implementation closeout

The Streamlit evidence-allocation dashboard is implemented entirely under `demo/`, on branch `gagan`. It was tested against the local primary artifacts, without model inference, training or changes to scientific results.

## What is available

- **Evidence Lab / Experimental Replay:** three datasets, both evidence budgets, selectable core methods, paired saved results, original/recovered unchanged answers and saved task scores, gold-reference expander, source panels and budget meters.
- **Mandatory Granularity Explorer:** source-exact passage/sentence decomposition; proposition views only for actually selected stored proposition IDs; exact H3 description; QBGF's registered fidelity options; observed per-source representation; cross-budget unit/source counts; explicit ranking-versus-granularity-versus-allocation explanation.
- **Evidence utility:** actual per-unit relevance and signed deletion utility; helpful/neutral/harmful labels; clicking a plot point focuses the corresponding evidence text.
- **Research Findings:** dynamically loaded evidence benefit, granularity/budget, query-conditioning and sentence-only fusion ablations, attribution summary and VAL-to-confirmation directional consistency. Simple and technical views keep task metrics separate.
- **Research Journey:** concise W10→W12 C/D→H/I freeze→K/L attribution→M confirmation narrative with primary-source links.
- Presentation/researcher controls, guided tour, primary provenance fingerprints, dark/light support and responsive sidebar/layout.

## Architecture and files

`app.py` orchestrates Curated Demo, LiveDemo, Research Findings and Research Journey. `data_loader.py` provides read-only schema-checked, cached primary loaders and manifest checks. `result_registry.py` resolves condition/query identity and deterministic curation. `prepare_replay.py` calls the existing canonical source adapters, hierarchy builder, pinned local tokenizer and serializer for CPU-only exact recovery. `components.py`, `plots.py` and `assets/styles.css` provide presentation components. `live_pipeline.py` provides the no-load/no-download availability gate. `constants.py` contains the registry/definitions. `artifact_manifest.json` records primary fingerprints. `example_catalog.json` records curated VAL IDs. `tests/`, `README.md`, `requirements*.txt` and `setup.sh` cover verification and operation. Generated environments, replay index, logs and screenshots are ignored.

## Primary data sources

See `artifact_manifest.json` for the exact paths and hashes. The main sources are:

- `results/week12/qbgf/v3/analysis/{summary,primary_comparisons,fusion_diagnostic_comparisons}.csv`
- `results/week12/final_confirmation/v1/{summary,contrasts,validation_comparison}.csv`
- `results/week12/attribution/v1/analysis/{unit_importance,query_profiles,relevance_utility}.csv`
- `results/week12/attribution/v1/analysis/qualitative_candidates.json`
- byte-identical Part-I freezes and split metadata; canonical algorithm/source hierarchy definitions
- W10 retrieval and W11 granularity CSVs for journey provenance

No historical Markdown is parsed as numerical evidence. Raw validation sources are used only to recover exact selected units; no confirmation question or label is loaded by the interactive lab.

## Replay verification and example curation

The CPU recovery produced **36 curated VAL queries, 12 per dataset**. Selected unit IDs, source substrings, individual token counts and complete serialized evidence-token totals matched the saved attribution exports. Original answers absent from retained qualitative JSON are shown only when an exact-unchanged ablation export establishes the original string; that provenance is visible. The derived index is about 6 MB, not a duplicate of large result/model files.

Curation is deterministic and explicitly outcome-based for storytelling. It prefers the retained real qualitative examples, then examples with larger observed paired profile variation. The query IDs are in `example_catalog.json`. Some curated questions show QBGF success, some failure, and some high relevance with neutral/harmful deletion effects. The app does not imply representative prevalence from this curated set.

## Important missing-artifact limits

Earlier cleanup removed detailed C/D prepared/result and attribution prepared shards. Some original top-10 pools can be fully recovered from saved parent metadata, while others are partial. Exact QBGF option labels/utilities, prompt/generated token counts, EM, and some method/query pairings cannot be recovered. They are labelled unavailable. No historical rank, text, metric, option or answer is silently synthesized.

The core method menu is available, but the app cannot invent a RAW_SKIP/FIXED_SENTENCE comparison for a query whose original paired rows no longer exist. Available H3/C2/BM25 comparison rows retain their true identities; BM25 is never presented as SFR. Aggregate findings still contain the full matched final comparisons.

Proposition decomposition is shown only where saved selected proposition membership exists. Canonically recovered unselected hierarchy spans are educational source views, not newly scored results. Observed per-source membership is not falsely relabelled as a surviving optimizer decision trace.

## Cached versus live

Curated Demo is CPU-only saved replay. LiveDemo is a separate custom-input mode:
BM25 context-only processing on CPU, optional local SFR retrieval and Qwen BASE_FP16
reader generation on GPU. Models load only on explicit runs, through a Streamlit
cached resource manager. SFR and Qwen never coexist as large resident GPU models;
GPU handoff/unload was manually qualified. Reader unavailability preserves exact
context inspection. No custom output enters historical artifacts or findings.
See the final LiveDemo section below for current verification details.

## Verification

- **55 automated tests passed:** primary loaders and values, exact method registry, metric mismatch/invalid conditions, missing results, deterministic VAL curation, missing cache, invalid query, source hashes/freeze, unavailable live models, matched retriever/reader/SARA slices, exact real packing costs, retriever scope and restoration after MiniLM fallback, no model modules on data import, all recovered selected-unit costs/text, Streamlit modes/datasets/budgets, pending-method state, technical mode, guided tour and an actual stored proposition view.
- **17 headless-Chrome browser checks passed:** desktop lab, actual scatter-to-source click, findings and each remaining finding tab, journey, tablet, narrow screen and dark theme. No app exceptions or detected page-width overflow. Screenshots were manually inspected; the narrow sidebar issue found during review was corrected.
- Browser widths: **1440, 820 and 520 px**. Browser tests use installed Chrome with GPU acceleration disabled; no browser/model download.
- All 29 registered primary source fingerprints and both final freeze SHA checks pass unchanged. `git status` shows additions under `demo/` only. Main was not modified.

Local QA receipts: `demo/artifacts/browser_checks.json`; logs `demo/all-tests.log`, `demo/browser-tests.log`. Screenshots are under ignored `demo/artifacts/` (`evidence_lab.png`, `granularity_explorer.png`, `evidence_utility.png`, `research_findings.png`, `research_journey.png`, `tablet.png`, `narrow.png`, `dark_lab.png`).

## Run

From the repository root:

```bash
bash demo/setup.sh
demo/.venv/bin/python -m streamlit run demo/app.py
```

The environment is already installed locally. A preview server is bound to `127.0.0.1:8501`; SSH users should forward that port. It does not bind publicly. For source recovery after a fresh checkout:

```bash
sara_env/bin/python -m demo.prepare_replay
```

This is CPU-only, using existing local tokenizer/data files. See README for requirements and the 90-second tour. Scientific experimental code, frozen splits, results, notebooks and checkpoints are unchanged.

## Retriever / reader / complete-pathway extension — 3 November 2026

Added `evidence_registry.py` and `research_extensions.py`, extending the primary registry
from 15 to 29 CSV/JSON sources while checking every previous fingerprint unchanged.
The hero now distinguishes Query → Retriever → Top-k pool → Granularity → Allocation
→ Reader → Answer → Utility analysis.

The lab includes a same-query BM25/MiniLM/SFR trace comparison and retriever-scoped
downstream replay. BM25/SFR parent metadata are partial for many examples; MiniLM
query traces do not survive cleanup. Changing to an unavailable retriever shows
retrieval-only/unavailable, never another retriever's answer. Switching back restores
valid method state; the browser caught and regression tests now cover that transition.
Shared-source labels refer only to observed traces. A definitive retrieval-miss case
cannot be established from partial pools and is not invented.

New findings cover three-retriever pilot recall/coverage, larger matched BM25/SFR
RAW_SKIP downstream controls, the same-checkpoint paired QASPER SARA specialists,
public-reader comparisons and format diagnostics, conceptual flexibility versus
quality, the frozen decision protocol, and retrospective gold-overlap/retention.
Task descriptions explain why the three datasets and metrics are not interchangeable.
The W11 recall and W12 quality samples are not falsely presented as one paired causal
retrieval→answer contrast. Reader interaction CIs remain exploratory and include zero;
SciFact public Mistral/Llama format failures are explicitly displayed.

The packing view uses exact standalone serialized block costs recovered using the
canonical code, not example numbers or rough word estimates. It explains whether a
real passage can fit before any other allocation; it does not reconstruct missing
optimizer choices. Selected-source counts and actual saved outcome comparisons remain
separate. All recovery used CPU tokenization only and wrote only the demo index.

QA: 55 automated tests and 17 browser checks passed. Desktop, narrow/tablet and dark
views were checked. New screenshots include `retriever_explorer.png`,
`packing_pathology.png`, `retriever_findings.png`, `reader_effects.png` and
`decision_protocol.png`. Exact rendered source/passage text and scores remain primary
artifact driven. No experiment, split, freeze, model checkpoint or notebook changed.

## Complete curated replay rebuild — 3 November 2026

The historical replay exports do not preserve complete source/answer/deletion data
for every lab combination. A new application-only rebuild workflow is implemented,
with no changes to experimental definitions or historical results.

Notebook: **`demo/notebooks/BUILD_CURATED_REPLAYS.ipynb`**. Set only
`RUN_FULL=True` and Run All to perform preparation, answers, deletion analysis and
publication. Defaults run a metadata plan plus synthetic smoke. Every stage resumes
immutable completed shards, checks identities and rejects overwrite. Retrieval and
reader generation use separate processes. No real run was started in implementation.

The immutable demonstration manifest selects **10 existing curated frozen VAL IDs
per dataset** (30 total), covers **SFR/BM25/MiniLM**, budgets **256/1024**, and nine
representations: NONE, RAW_SKIP, FIXED_SENTENCE, C2, H3, QBGF_FULL, PROPOSITION,
MULTI_SENTENCE_2 and MULTI_SENTENCE_4. The first six call the original canonical code;
the remaining three are explicitly labelled exact-source demo-only policies.
All 1,620 requested answer conditions must complete. The utility explorer additionally
requires actual deletion results for every selected unit. No missing importance is
filled with zero, and no other retriever/method's answer is substituted.

A published `demo/replay_data/v1/bundle.json` contains every selected question,
reference, complete retrieval pool, source hierarchy, selected context, generated
answer, metric/cost, optimizer choices and observed unit deletion. Its SHA receipt is
`PUBLISHED.json`. It no longer depends on old detailed experimental or embedding
caches to render the lab. These are **durable data**, not disposable caches; back up
the bundle, receipt and manifest separately. During initial development all of `/demo/` was temporarily ignored
by the repository `.gitignore` as requested. Unfreezing it later requires removing
that root ignore entry; nothing was removed from disk.

The app prefers the complete collection automatically once published. Before that,
it states that historical exports are incomplete and points to the notebook. Its
original research findings stay primary-artifact driven and are not replaced by
curated-run averages. A stale historical source index no longer disables a complete
new collection. NONE/empty contexts no longer enter an empty unit-selection widget.

Verification: **66 demo tests passed**, including **11 new curated replay tests**;
**17 browser checks passed**. New tests cover complete grid identity, all nine source
representation policies, immutable resume, exact-prompt generation deduplication,
actual generation/deletion plumbing through a fake reader, missing-answer/missing-LOO
publication rejection, repeat publication, checksum tampering, self-contained loading
after intermediate shard removal, all three dataset UI views and NONE handling.
The notebook's default Run All was executed successfully. All 30 actual VAL sources
and references also loaded in a CPU-only source check, including all 5,183 SciFact
corpus documents; no embeddings or real answers were generated. Logs:
`demo/all-tests.log`, `demo/browser-tests.log`, `demo/curated-notebook-smoke.log`,
`demo/curated-source-check.log`.

The real 1,620-row/deletion collection is **pending manual execution**. Final source
models and frozen-reader execution are implemented but were not exercised with real
GPU inference in this patch. Unit deletions can dominate runtime; their exact count
is printed after preparation. No historical QASPER TEST or confirmation cohort was
accessed. All new model work remains disabled by default.

## LiveDemo and polished curated presentation — current update

### Files and behavior

- `app.py`: four-mode navigation — Curated Demo, LiveDemo, Research Findings,
  Research Journey. Historical export selection is confined to Researcher Mode.
- `curated/ui.py`: clean “Curated Research Examples” introduction. Collection counts,
  curation/benchmark limitations and internal hashes remain in Researcher-only
  “About this demo”/provenance panels. Saved results and metadata were not edited.
- `live/engine.py`: validated custom-input pipeline; actual query/source processing,
  BM25/SFR ranking, exact sentences, canonical NONE/RAW_SKIP/FIXED_SENTENCE/QBGF_FULL,
  exact allocation traces, source-order serialization and deterministic explanations.
- `live/resources.py`: lazy pinned local tokenizer/SFR/Qwen resources, one large
  GPU model at a time, cached through Streamlit, with a shared execution lock.
  Qwen/SFR switching accounts for reusable same-process allocator memory; unload
  moves weights off GPU and clears retained CUDA workspaces. No downloads.
- `live/ui.py`: two-column passage/query inputs; add/remove/import (max20); budget
  and method controls; actual stage previews and persistent processing indicator;
  candidate pool, exact context, source/sentence representation, answer and costs;
  same-input method comparison and conditional BM25/SFR comparison.
- `assets/live_tokenizer/`: about15MiB of pinned tokenizer-only assets with a source
  pin and file checksums. Functional token IDs/chat template match the project’s
  cached Qwen tokenizer. This supports CPU context inspection without reader weights.
- `assets/styles.css`: stage cards use the existing colors, typography and evidence
  language. Processing animation reflects actual status only; reduced-motion is
  respected. No artificial waiting is introduced.
- `tests/test_live.py` plus updated application/browser tests and README.

### Supported live paths

| Path | Availability / behavior |
|---|---|
| BM25 | CPU ranking and exact evidence allocation |
| SFR | Offered when pinned local weights and GPU capacity are available; real qualified path |
| Reader | Qwen3-8B BASE_FP16; explicit Enable Live Models, loaded on the first requested answer |
| NONE | Retrieval/granularity skipped; zero evidence, query-only generation |
| RAW_SKIP | Actual ranked packing decisions show inclusion, oversized skips and remaining budget |
| FIXED_SENTENCE | Exact-source sentence scores, skip-to-fit packing and source-order restoration |
| QBGF_FULL | Actual shared-budget DROP/SENT_k/FULL choices and relevance-surrogate utility |
| Missing/disabled reader | Exact context, rankings, units, packing and explanation remain available; no invented answer |

Custom queries never have F1, EM, gold-answer or correctness cards. Runtime
explanations are assembled directly from actual metadata. Hidden-thinking blocks
are suppressed rather than displayed. Custom content/results remain in session
memory; bounded authored QA receipts/screenshots are only under `demo/runtime/qa/`.
No validation/confirmation/benchmark result file is opened for writing.

### Qualification

**96 automated tests passed**, including29 LiveDemo tests: input parsing/validation,
all four methods and both budgets, deterministic retrieval/ties, exact serialization,
RAW_SKIP decision reconstruction, QBGF choices, NONE, missing/disabled reader,
comparison identity/cache behavior, source-grounded units, no hidden-thinking output,
bundled tokenizer independent of the model cache, GPU allocator accounting and
interactive UI cases. The test for the old build notebook now checks execution guards
without changing the user’s already-edited RUN_FULL/RUN_SMOKE flags.

Browser receipts cover17 existing-page checks,10 curated/live context-only checks,
and5 real-model checks: BM25+Qwen, SFR+QBGF+Qwen, same-input method comparison,
BM25/SFR ranking comparison and unload. A separate real-token packing check used five
source passages, skipped the oversized second passage and included P1/P3 at242/256
actual evidence tokens. Seven small authored backend conditions cover NONE with three
passages; RAW_SKIP/FIXED_SENTENCE@256; FIXED_SENTENCE@1024; QBGF; disabled-reader
fallback; and SFR/QBGF/reader handoff. Six of those conditions produced actual local
Qwen answers; one intentionally used context-only mode. These are plumbing tests,
not research evaluations or latency benchmarks.

The full reader/encoder models were unloaded after qualification. A measured unload
returned Torch allocated/reserved model memory to0MiB; the normal CUDA process context
can remain. No expensive research run, model download or historical experiment change
was performed. The existing user-facing server was left alone; QA used a separate
local port8502.

Screenshots/receipts: `runtime/qa/curated_clean.png`, `live_inputs.png`,
`live_real_answer.png`, `live_real_sfr.png`, `live_comparison.png`, `live_narrow.png`,
`live_model_cases.json`, `browser_live_checks.json`, `browser_live_models.json` and
`gpu_release_verified.log`.

### Launch

Restart the application to clear old imported modules/resources, then run:

```bash
demo/.venv/bin/python -m streamlit run demo/app.py
```

Open LiveDemo. Start with Try a sample and CPU context-only mode; enable Live Models
for an answer. Curated Demo remains CPU-only and unchanged numerically. Current live
GPU qualification is specific to the local NVIDIA TITAN RTX; other environments may
correctly use the context-only fallback.

A final real-browser check also verified that the processing bar stays at the top
while scrolling during actual Qwen loading (`persistent_progress_check.json`,
`live_processing.png`). Together with the other receipts this gives33 browser checks.
Final standalone model, CPU tokenizer and exact-packing checks passed. The small
bundled tokenizer allows exact CPU allocation even without an external Qwen weight
cache; it contains no model weights and was copied locally, not downloaded.

## Final freeze verification

`evidence-lab-demo-1.0.0` passed all 149 demo tests. An isolated Git-index export restored the immutable curated archive and launched Curated Demo with no Torch import or model inference. The compressed research split metadata was read without writing an uncompressed research file. Controlled retriever-only comparison is the presentation default; native comparison and historical lexical QBGF are researcher-only. The static budget-sensitive preset exposes actual P7/P9 allocation states. All 771 protected files remain unchanged. Source and the versioned compressed archive are preserved on branch gagan; runtime data and raw bundle stay selectively ignored. No further algorithm or research experiment was added.

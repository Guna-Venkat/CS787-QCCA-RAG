# Evidence Lab · CS787 research demo

An interactive **Empirical Study of Budget-Constrained Evidence Allocation in RAG**.
The main visual object is source evidence and its allocation—not a chat conversation.
Research Findings reads numerical results from primary CSV/JSON artifacts. LiveDemo reports only actual runtime metadata for custom inputs. Historical Markdown is
not used as numerical evidence.

## Complete user and method guide

See [APPLICATION_GUIDE.md](APPLICATION_GUIDE.md) for all selectable methods, exact budget semantics, live execution, saved-example interpretation, metrics and application panels.

## Final frozen release

Version **evidence-lab-demo-1.0.0**. Presentation comparisons default to controlled SFR downstream scoring. Advanced/native variants are researcher-only. See [FINAL_FREEZE_STATUS.md](FINAL_FREEZE_STATUS.md).

A clean checkout includes a versioned 13.93 MiB archive, not a raw 98 MiB JSON object. Restore the original curated bundle without model inference:

```bash
python3.11 -m venv demo/.venv
demo/.venv/bin/python -m pip install -r demo/requirements.txt
demo/.venv/bin/python demo/restore_curated_bundle.py
demo/.venv/bin/python -m streamlit run demo/app.py
```

See [archive integrity and restore instructions](releases/README.md). Existing environments can skip creation/installation. Source, tests, UI assets, audit documentation and tokenizer-only assets are now selectively versioned; runtime/private caches and weights stay excluded.

## Launch

From the repository root, on `gagan`:

```bash
bash demo/setup.sh
demo/.venv/bin/python -m streamlit run demo/app.py
```

Open the local URL printed by Streamlit. For an SSH server, forward port 8501 through
your normal SSH/VS Code port forwarding. The app remains bound to the local interface
unless you explicitly configure otherwise. No tokens or model access credentials are
requested by the demo.

### Recover source views (CPU only, no model generation)

A local verified index was prepared during implementation. If it is missing, the app
still renders all research findings, numerical attribution profiles and surviving
qualitative text. Recreate richer source views with:

```bash
sara_env/bin/python -m demo.prepare_replay
```

This imports the project's existing source adapters, hierarchy builder, pinned local
Qwen tokenizer and serializer. It verifies **exact unit IDs, source substrings,
standalone token counts and complete selected-evidence costs** against the saved
validation attribution exports. It does not retrieve, recompute semantic scores,
rerun selection, generate answers, access confirmation examples, or write experiment
results. Its small derived index lives only in ignored `demo/cache/`.

Requirements for recovery: the existing local raw VAL data, corpus files and pinned
Qwen tokenizer. Recovery never downloads them. The normal cached dashboard does not load
torch, transformers, a tokenizer or a GPU.

## The presentation

1. **Evidence Lab:** choose a real VAL query and budget; compare recoverable BM25/SFR
   candidate traces, select the retriever, inspect paired available methods, meters, answers and evidence. MiniLM has aggregate results but no retained query replay.
2. **Granularity Explorer:** open a source, see its full passage and exact sentences,
   and selected proposition artifacts when genuinely present. The conceptual ladder
   is not a quality ranking. Compare saved source/unit counts at both budgets.
3. **Evidence Utility:** select a scatter point to highlight its text. Color indicates
   helpful/neutral/harmful under deletion, separate from selection status.
4. **Research Findings:** evidence benefit, granularity/budget, query conditioning,
   fusion ablation, relevance/utility, retriever effects, reader effects, complexity and
   the VAL→freeze→confirmation decision protocol. Gold-overlap and finite-retention
   diagnostics are available with their retrospective/eligibility caveats.
5. **Research Journey:** a concise W10→W12 narrative and explicit claim boundaries.

Presentation Mode is on by default. Researcher Mode exposes larger exact tables.
Every major result panel includes relative source paths and content fingerprints.

### 90-second guided route

Enable the tour, choose SciFact, and compare retained retrieval traces. Freeze SFR
for the example, open a source, and inspect the granularity views. Toggle 256/1024
and open the exact-cost packing panel. Compare available methods, QBGF fidelity and
delete-one utility. Finish in Findings with retrieval, reader, allocation ablations
and the frozen final confirmation. Unavailable traces/results remain labelled.

Examples are **outcome-curated validation illustrations**, not a representative new
sample or confirmation-based winner selection. The selector is deterministic.

## Scientific scope and limitations

- Core method definitions: `src/week12/algorithm.py`; no copied/reimplemented selector.
- The final primary policy is FIXED_SENTENCE, not a claim of universal best quality.
- All three datasets have aggregate findings and saved VAL attribution profiles.
- 36 curated queries (12/dataset) have source-exact recovered hierarchies locally.
- Detailed original C/D prepared/result shards and attribution prepared shards were
  removed in earlier cleanup. Consequently full top-10 historical retrieval traces,
  exact QBGF DROP/SENT/FULL choices/utilities, per-query prompt/generated-token counts
  and some method/query pairings **cannot be replayed**. Missing values are labelled,
  never invented or inferred from a new run.
- Known retrieved parents come from saved selected-unit parent ranks/scores. They are
  not presented as the complete top-10 pool. BM25 ranks are separate from SFR ranks.
- Original answers not directly retained can be recovered from an ablation whose
  saved `exact_flip=False` proves its answer unchanged. That provenance is visible.
- The educational hierarchy is recovered from canonical source code and matched IDs;
  it is not a newly evaluated alternative context. Proposition views require a saved
  selected proposition ID for that query/source.
- RAW_SKIP/FIXED_SENTENCE/H3/QBGF/NONE remain selectable; unavailable paired query rows
  show pending/unavailable, not a fabricated answer. Available historical C2/BM25
  comparator rows are explicitly labelled with their actual identities.
- Per-example SciFact shows correctness; macro-F1 is aggregate only. Metrics are never
  averaged across datasets.
- Neutral deletion does not prove evidence non-use. Input ablation is not internal
  causal reasoning. Nominal intervals and adjusted p-values have separate meanings.
- Confirmation is shown only in aggregate. It is now consumed, not available for tuning.

## Curated Demo versus LiveDemo

Navigation: **Curated Demo · LiveDemo · Research Findings · Research Journey**.

Curated Demo displays the saved collection on CPU without loading inference models.
Its presentation view has a clean “Curated Research Examples” introduction. Counts,
curation limitations, collection hashes and internal provenance remain available in
Researcher Mode; the underlying bundle is unchanged. Historical exports are also
available only in Researcher Mode.

LiveDemo takes up to 20 custom passages plus a query. Add/remove passage cards or
paste blocks separated by a line containing `---`. Choose budget256/1024 and NONE,
RAW_SKIP, FIXED_SENTENCE or QBGF_FULL, then press **Run Evidence Pipeline**.

- **BM25 context-only mode works on CPU**, with exact local Qwen tokenization. It
  still shows ranking, sentence decomposition, actual packing and exact evidence.
- Enable **Live Models** to request Qwen3-8B BASE_FP16 answers. SFR is offered only
  when its pinned local checkpoint and sufficient GPU memory are available.
- No model loads on application startup. A Streamlit cached resource manager loads
  models on request, serializes GPU operations, and releases SFR before Qwen loads.
  The unload button also clears retained CUDA workspaces so memory returns to other
  applications. No model is downloaded.
- NONE skips retrieval and granularity and prepares a query-only reader prompt.
- Stage indicators reflect real operations, with no fake delays or hidden reasoning.
  Detailed previews show actual rankings, sentence scores and allocation decisions.
- Reader failure or disabled models retain the evidence view with a clear unavailable
  message. The small exact tokenizer is bundled under `assets/live_tokenizer/`, so CPU
  context processing does not require Qwen weights or its external model cache.
  The pin and file hashes are checked before use; no approximate evidence budget
  is silently substituted.
- “Compare with another method” preserves the original inputs, budget and retriever;
  completed session results and prepared sources are reused. “Compare Retrievers”
  appears only when SFR can run, and does not request a reader generation.
- Custom queries have no gold F1, EM or correctness scores. Runtime explanations are
  deterministic descriptions of actual operations, not LLM-generated reasoning.
- Live inputs/results stay in session memory. Bounded qualification receipts and
  screenshots use `demo/runtime/qa/`. No custom output enters research CSVs, frozen
  artifacts, benchmark summaries or the curated collection.

A small authored sensor example is available via **Try a sample**. Live model paths
were qualified with real local Qwen/SFR weights on this server. The first model run
includes loading time; switching large models also costs time. These observations
are not serving benchmarks. Leave Live Models off for CPU-only evidence inspection.

## Architecture

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit modes, paired replay, granularity and story |
| `data_loader.py` | Read-only primary data, schema checks, cached parsers, hashes |
| `result_registry.py` | Dataset/method/budget/query lookup, deterministic curation |
| `evidence_registry.py` | Matched retrieval/reader/specialist slices, trace scope and packing facts |
| `research_extensions.py` | Retriever explorer, reader effects, packing, complexity, decisions and case studies |
| `prepare_replay.py` | Optional source-exact CPU recovery, using canonical code |
| `components.py` | Cards, meters, source panels, source text escaping |
| `plots.py` | Interactive Plotly results and unit selection |
| `live/engine.py` | Validated custom-input evidence pipeline, exact packing trace and deterministic explanations |
| `live/resources.py` | Lazy local-only model resources, GPU handoff and unload |
| `live/ui.py` | Live inputs, progress, evidence/result cards and comparisons |
| `live_pipeline.py` | Legacy historical availability utility; not the current LiveDemo backend |
| `constants.py` | Registries, method explanations, source paths |
| `assets/styles.css` | Responsive dashboard visual design |
| `tests/` | Primary-value, identity, fallback and model-free UI tests |

Primary inputs are listed in `artifact_manifest.json`. The manifest contains metadata
and hashes only, not duplicated result files. No numerical result comes from reports.
Large experiments/models remain outside `demo/`; generated index, environment and
screenshots are ignored.

## Tests

```bash
sara_env/bin/python -m pytest demo/tests/test_registry.py -q
# After installing Streamlit:
demo/.venv/bin/python -m pytest demo/tests/test_registry.py demo/tests/test_extensions.py demo/tests/test_app.py -q
```

Browser smoke uses installed Chrome with Playwright (`requirements-test.txt`), not a
new browser/model download. Screenshots and QA receipts are local under ignored
`demo/artifacts/`. No primary scientific code, split, result or freeze is changed.

## Retriever, reader and broader-story additions

- **Retriever Effect Explorer:** BM25/MiniLM/SFR controls, exact pinned model IDs,
  same-query retained-parent columns, observed shared sources and retrospective
  overlap indicators. Partial traces never imply unique retrieval or a definite miss.
- **Retrieval ceiling:** matched Week11 P0 any/all/coverage curves by dataset, unit
  and cutoff. The pilot has its own cohort and denominator.
- **Matched downstream retrieval:** Week12 C-EXT BM25/SFR with the same Qwen reader,
  RAW_SKIP policy and budget. Do not causally pair this with Week11 pilot recall.
- **SARA case study:** paired same-checkpoint k5/n10 QASPER specialists, separated
  from public Qwen systems. Historical specialist results do not prove latent
  superiority or a universally best retriever.
- **Reader effects:** public Mistral/Qwen/Llama under the Week11 shared-selection
  protocol, with SciFact invalid-format rates and exploratory interaction CIs.
  Exact per-query source hashes cannot be re-audited after shard cleanup. The
  Week10 public panel is separately labelled; adapted C-Mistral is not mixed in.
- **Packing:** a real recovered source passage is measured by the existing exact
  serializer and local tokenizer during CPU recovery. The graph checks whether
  that full source can fit in an empty 256/1024 budget. It is not an inferred
  historical RAW_SKIP trace or a guarantee that passage selection improves quality.
- **Failure/utility cases:** actual retained qualitative exports only. No complete
  paired three-retriever miss case survives, so that case remains unavailable.
- **Decision protocol:** VAL comparisons, predefined promotion gate, fallback freeze
  and reserved confirmation stay separate. Complexity is conceptual algorithm
  flexibility, not an invented runtime/parameter-count axis.

All changes remain under `demo/`. No full experiment, retrieval generation, reader
inference or model training was launched.

## Rebuild a complete end-to-end Evidence Lab collection

The retained historical exports do **not** contain an answer for every interactive
combination. Source recovery alone cannot repair missing generated answers. A new,
separately labelled, self-contained collection is now implemented.

Open **`demo/notebooks/BUILD_CURATED_REPLAYS.ipynb`** using `sara_env`.

| Setting | Value |
|---|---|
| First safe check | Keep `RUN_FULL=False`, `RUN_SMOKE=True`; Run All |
| Build the complete collection | Set `RUN_FULL=True`; Run All |
| Resume after interruption | Keep `RESUME=True`, `OVERWRITE=False`; Run All again |
| Stage switches | Keep preparation, answers, attribution and publication all `True` |

The matrix is **10 frozen VAL examples per dataset × 3 retrievers × 9 representations
× 2 budgets = 1,620 saved answer rows**, plus actual leave-one-out unit-deletion
results. Only VAL membership is used; no confirmation cohort or historical QASPER
TEST is accessed. Existing examples were outcome-curated for explanation; they are
not a representative evaluation sample. No model is selected or tuned using the new
outputs.

Retrievers: SFR, BM25, MiniLM. Reader: the existing pinned Qwen BASE_FP16, no LoRA.
Core methods: NONE, RAW_SKIP, FIXED_SENTENCE, C2, H3 and QBGF_FULL, calling the original
canonical implementations. Additional explicitly demo-only exact-source views:
PROPOSITION, MULTI_SENTENCE_2 and MULTI_SENTENCE_4. The latter rank sentences within
each source by the retriever's fine-unit score, keep up to two/four, and include
fitting passage blocks in retrieved order. BM25 fine scores are lexical; dense ones
are cosine. No generated extraction or synthetic source text is used.

The notebook runs retrieval/preparation and answer generation in **separate
processes**, freeing SFR before Qwen loads. All writes, embedding caches, source
hierarchies, selected contexts, answers, exact cost counters and deletion results
stay under `demo/replay_data/v1/`. No historical study file, freeze or split is
changed. The final release selectively versions source and the immutable compressed bundle archive; runtime/intermediate data remain ignored.

Publication is strict: every requested combination must have its actual answer and
every selected unit its actual deletion result. There are no invented zero utilities
or substituted answers. A partial run remains resumable but is not advertised as a
complete collection. After preparation, the plan displays the exact number of unit
ablations; these can dominate runtime. Identical query/evidence prompts reuse exact
answers while retaining conceptual one-call cost. No generation begins with the
notebook's defaults. The published curated grid is complete. The final freeze uses its verified archive; rebuilding it is unnecessary.

Once complete, publication writes:

- **`demo/replay_data/v1/bundle.json`** — the portable collection, including full
  sources, retrieval pools, answers, metrics, allocation choices and deletion data.
- **`demo/replay_data/v1/PUBLISHED.json`** — checksum and complete-grid receipt.
- **`demo/replay_data/v1/manifest.json`** — immutable curated IDs/config/source identity.

**These three files are durable demo data, not disposable caches. Back them up
as the versioned immutable archive; the raw JSON is restored locally and remains ignored.** The published bundle no longer
needs intermediate model embeddings or old replay caches to render the lab. The
Research Findings panels still use their original audited aggregate sources.

Restart the app after publication. Evidence Lab automatically defaults to
**Complete curated demo**; the older historical-export view remains available and
clearly separated. Each question/retriever/budget/method option then has a real saved
end-to-end result. Research Findings continues to show the frozen study, not new
demo averages.

CLI alternative:

```bash
sara_env/bin/python demo/build_curated_replays.py plan
sara_env/bin/python demo/build_curated_replays.py smoke
# Expensive, explicit stages; rerunning resumes each one:
sara_env/bin/python demo/build_curated_replays.py prepare --full
sara_env/bin/python demo/build_curated_replays.py answers --full
sara_env/bin/python demo/build_curated_replays.py attribution --full
sara_env/bin/python demo/build_curated_replays.py publish --full
```

### LiveDemo controls and privacy

Use **Try a sample** for a small authored illustration, or replace every passage
with your own text. Empty passage cards must be filled or removed. Input limits are
20 passages, 8,000 characters per passage, 30,000 total source characters and 512
reader tokens for the query. Evidence budgets use exact tokenizer IDs, not the
word-based estimate displayed beside the input cards. User passages remain intact
as native candidate units, so RAW_SKIP can visibly skip an oversized passage.

BM25 uses lexical scores for passage and sentence ranking; SFR uses semantic cosine.
Both use the canonical source-order serializer and frozen allocation logic. The
same-input comparison cache cannot substitute another query, source set, retriever
or budget. “Exact Evidence Prepared” is used when no reader call happened; the UI
does not claim that an unavailable reader consumed the context.

LiveDemo does not compute gold-reference metrics or leave-one-out utility for custom
queries. It shows actual selected/dropped spans, packing decisions and model output.
Source cards, token meters and answer typography are shared with Curated Demo.

### CPU FP16 LiveDemo (sequential loading)

Restart Streamlit after updating the application. In **LiveDemo**, choose
**CPU · FP16 · sequential models** (the default), enable **Live Models**, and
select **SFR** if semantic retrieval is wanted. Press **Run Evidence Pipeline**.
Only local pinned checkpoints are used; no downloads are triggered.

SFR query/passage/sentence encoding runs on CPU in FP16. Its model is discarded
before Qwen3-8B is loaded on CPU in FP16. After answer generation, Qwen is also
unloaded. Small reusable embeddings and the tokenizer remain cached. Freed tensor
memory is returned to the OS where supported; reclaimable filesystem cache may
remain visible. Context-only runs also unload the encoder at completion.

The reader retains the frozen prompt, token-ID evidence, parser, greedy decoding
and generation cap. CPU uses supported SDPA attention rather than the CUDA-only
attention backend. CPU numerical outputs are demo outputs, not replacements for
saved research results. Eight CPU threads are used, with a 24,000 MiB available-RAM
check before each large-model load. GPU mode remains available explicitly.

A real two-passage CPU-only FP16 check on this server returned **“Sensor B lasts
longer.”** in **25.8 seconds**, including sequential model loading. Peak process
resident memory was **18.2 GiB**; both models were unloaded at completion.
This tiny example is a functional check, not a general latency estimate.

### Live sentence segmentation audit

LiveDemo offers explicit **Citation / Markdown-aware** and **Frozen research splitter** modes. The former reuses the frozen academic utility with a live-only, offset-preserving citation/Markdown boundary adapter; the latter preserves the original citation limitations exactly. Old session results are separated by segmentation identity. Open **Sentence segmentation audit** in a completed run to inspect/export all raw passages and source units. Read the [complete World Cup audit](audit/live_sentence_segmentation_audit.md). Restart Streamlit after this update.

### Final methodology corrections

See the [final correctness audit](audit/FINAL_DEMO_CORRECTNESS_AUDIT.md), [historical QASPER limitation audit](audit/qasper_frozen_splitter_impact.md), and [reproducibility status](audit/demo_reproducibility_status.md). LiveDemo now distinguishes pipeline-native from controlled retriever-only comparisons. Controlled mode uses common SFR downstream scoring after either candidate retriever. New BM25-native QBGF execution is blocked because lexical scores do not share the cosine softmax scale. The end-to-end controlled view freezes downstream settings and exposes exact contexts and answers without a custom gold score. Old protocol results cannot be reused; available legacy session records are retained with obsolete-protocol labels. Frozen research outputs and curated answers remain unchanged.

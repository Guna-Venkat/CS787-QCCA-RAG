# MiniLM reranker implementation and bounded smoke

**POST_W12_RERANKER_BASELINE_EXPLORATORY**. Branch: **gagan**. Full reader study was not started.

## What is ready

Notebook: `W12-POST/BaselineComparison_MiniLM.ipynb`. Set **RUN_FULL=True**, then **Run All**. Defaults leave full generation, scoring smoke, reader smoke and dependency installation disabled. The small local MiniLM model is already available; compatible scores/answers resume without regeneration. No global package changes were needed.

Implementation: `W12-POST/reranker_baseline/{protocol,minilm_reranker,runner,analysis}.py`, `tests/`, `README.md`, and `W12-POST/run_minilm_reranker.py`. All new outputs, local model weights and protocol/cache/row/report writes are scoped to `W12-POST/results/minilm_reranker/`. A local .gitignore excludes weights/intermediate caches and test scratch files. Frozen source files outside W12-POST were read-only.

## Model and allocation

`cross-encoder/ms-marco-MiniLM-L6-v2` @ `233902d25c440f23af6f7d6e94d2946bac0bee0a`; **cpu / torch.float32**; **22,713,601 parameters**. Raw logits; eval mode; no training; max pair length 512. Local model footprint approximately 88 MiB. Official source: https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2 .

Query/sentence pairs only; no gold, F1, LOO or P_help enters scoring. The cached frozen SFR top-10 source sentences are reused exactly. Ties use passage rank, source offset and stable sentence ID. Fit/skip packing uses the unchanged exact serializer, exhausts candidates even for negative raw scores, and restores original source order. The reader remains the exact pinned canonical Qwen3-8B BASE_FP16; it is loaded only by an explicit manual reader run.

## Checks completed

**18 unit tests passed.** They cover ranking/ties, exact packing, oversize skip-and-continue, negative-score inclusion, original-order presentation, no label leakage, duplicate/source mismatch rejection, budget registry, atomic immutable resume, incompatible identities, out-of-scope writes, full-run unlocks, bounded smoke membership, five-method tables/ties, wrong retrieval pool, Unicode hash conventions, and synthetic cached-reader resume without loading a reader. The synthetic reader test uses test-only files and no real answers.

**Three real cross-encoder-only smoke queries; zero Qwen calls.** Both budgets were checked using one score pass/query. No scored pair was truncated. No smoke quality/F1 conclusion is drawn.

| Dataset | Query ID | Evidence @256 | Evidence @1024 | Sentence pairs scored |
|---|---|---:|---:|---:|
| HotpotQA | `5a7100435542994082a3e4a3` | 256 | 1024 | 64 |
| QASPER | `01a41c0a4a7365cd37d28690735114f2ff5229f2` | 255 | 1024 | 87 |
| SciFact | `1012` | 253 | 1022 | 130 |


**Integrity:** all **1212** baseline-row/protected-input file hashes match the saved protocol. Exact membership across all 1,200 control rows was verified before smoke. All implementation hashes match the smoke protocol. All selected smoke texts exactly match scored source strings; score ordering, canonical evidence counts and source-order presentation pass. The notebook remains unexecuted.

A pre-model smoke attempt initially caught a new provenance-guard mismatch: retrieval-lineage hashes use Week-12 JSON encoding while unit hashes use Week-11 Unicode encoding. The guard was corrected to check each original convention independently. No historical hashes were changed. That failed attempt, before any model scoring, is retained under `smoke/failed_attempts/pre_model_hash_guard/`. The subsequent three-query smoke completed successfully.

## Smoke outputs

`smoke/SMOKE_REPORT.json` contains all scores, top-10 ranking, include/skip costs and decisions, original-order evidence, and model metadata. `smoke/SMOKE_REPORT.md` is the readable companion. `protocol/MODEL_FILES.json` records pinned local model file hashes. Three score caches can be reused for the manual full run.

## Full-run outputs after manual execution

Expected 300 MiniLM answer rows (150 development queries × two budgets), combined read-only with four historical controls into 30 summary cells. Reports include the exact six-row five-method comparison, secondary EM/accuracy, efficiency, descriptive point-estimate wins, and 24 paired comparisons using the existing grouped 10,000-resample/seed-42 statistics with one Holm family of 24. No pooled dataset score or automatic promotion/expansion.

```bash
PYTHONDONTWRITEBYTECODE=1 sara_env/bin/python W12-POST/run_minilm_reranker.py full --execute
```

Expected final report: `W12-POST/results/minilm_reranker/reports/FINAL_COMPARISON.md`. The official method remains SFR top-10 → FIXED_SENTENCE → Qwen3-8B BASE_FP16 → no LoRA. The BEA_V2 calibration remains CALIBRATION_FAILED_TERMINATED.

**FULL 300-ANSWER RUN WAS NOT STARTED. NO REAL READER GENERATION EXECUTED.**

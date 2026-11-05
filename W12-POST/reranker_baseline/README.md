# MiniLM cross-encoder matched-cohort baseline

**POST_W12_RERANKER_BASELINE_EXPLORATORY**. This is a POST-W12 exploratory matched-cohort comparison on the same previously analyzed 50 development queries/dataset used by BEA_V2. The official final method and failed BEA_V2 calibration remain unchanged.

## Run manually

Open `W12-POST/BaselineComparison_MiniLM.ipynb` in `sara_env`. Set **`RUN_FULL=True`**, then **Run All**. Keep resume enabled and overwrite disabled. CPU reranking finishes and releases MiniLM before canonical Qwen GPU generation begins. If interrupted, rerun the same notebook; completed immutable score/selection/answer rows are reused. The final section automatically shows the six-cell, five-method comparison, efficiency, win counts and paired contrasts after all 300 MiniLM rows complete.

Defaults disable full inference, smoke scoring, reader smoke and dependency installation. Model availability is checked, but no weights are loaded merely by opening the notebook. The implementation smoke primes the local model and three score caches, so no additional model download should be needed. If the model cache is later removed, explicitly enable `ALLOW_MODEL_DOWNLOAD=True`.

Equivalent command from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 sara_env/bin/python W12-POST/run_minilm_reranker.py full --execute
```

Model/scoring-only smoke (exactly 3 development queries, one/dataset):

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 sara_env/bin/python W12-POST/run_minilm_reranker.py smoke --execute --allow-model-download
```

Qwen is not loaded by that smoke command. A separate manual `--reader-smoke` option exists for at most three answers @256; it is disabled by default and is not part of the agent's requested cross-encoder-only smoke. Smoke quality is not interpreted scientifically.

## Frozen and new identities

- Reranker: `cross-encoder/ms-marco-MiniLM-L6-v2` at `233902d25c440f23af6f7d6e94d2946bac0bee0a`; raw single-label logits, FP32 CPU, evaluation mode, pair max length 512, batch 32. Pre-truncation lengths and truncation flags are recorded; truncation affects scoring only and never rewrites source evidence.
- SFR top-10 pool and exact cached source sentences are loaded read-only. No retrieval, segmentation, encoder features, predictor training, threshold calibration or controller is added.
- Top-10 is the configured retrieval ceiling. If the frozen upstream artifact contains fewer than ten passages, reuse that exact nonempty pool without padding, re-retrieval or dropping the query. Preflight lists these cases, and answer rows distinguish configured `retrieval_width=10` from actual `retrieved_passage_count`. Candidate IDs, source hashes and all four control-source identities must still match exactly.
- Scores depend solely on `(query, sentence)`. Descending score ties resolve by passage rank, source position, stable sentence ID. Negative scores are still candidates; no score threshold is imposed.
- Greedy fit/skip uses the unchanged Week-12 exact serializer and restores source order. Budgets remain 256/1024. Incremental costs are measured by trial serialization, not by summing standalone sentence token counts.
- Same pinned Qwen3-8B BASE_FP16, prompt, parser, decoding, evaluator and no LoRA. No reader substitution is allowed. A free GPU with ≥18 GiB headroom and ≥3 GiB disk headroom is checked when a missing reader answer needs generation.
- Existing FIXED/QBGF/V1/V2 answers are never regenerated. All four methods must exactly match each other and the frozen cohort/group/source identities before proceeding. Their exact-prompt answers may be borrowed for a MiniLM row only when its source/prompt/model identities match.

## Outputs and comparison

All new code, notebook, weights, protocol, scores, selections, reader rows, receipts, tests and reports stay under **W12-POST/**. Outputs are `W12-POST/results/minilm_reranker/`:

- `protocol/`: immutable identity, model file hashes, derived progress status.
- `model_cache/`: only the selected official MiniLM checkpoint files (no Qwen/SFR copy).
- `cache/scores/`: one immutable per-query sentence score cache shared by both budgets.
- `smoke/`: bounded-query smoke report, allocations and optional reader rows.
- `full/selections/`, `rows/`, `cache/answers/`, `full/receipts/`: durable full-study results, created only on manual execution.
- `analysis/`, `reports/FINAL_COMPARISON.md`: 30 method/dataset/budget cells, 24 paired MiniLM contrasts, the final table, efficiency table, descriptive cell wins, SciFact class diagnostics and conservative interpretation.

Primary QA metric: answer F1; SciFact: grouped-resampled three-class macro-F1. Secondary QA EM and SciFact accuracy are retained. Paired cluster-bootstrap CIs and cluster label-swap raw p-values use the existing machinery, 10,000 resamples, seed 42; Holm adjustment is one exploratory family of all 24 contrasts. QASPER paper and SciFact cited-document grouping are unchanged. No pooled score. Cell winners are point estimates, not statistical victories. Exact ties credit all tied methods.

Gold retention and redundancy are retrospective diagnostics only; they are computed after evidence selection. No gold, F1, LOO, aspects or P_help enters ranking or packing. Missing annotations/undefined redundancy remain unavailable. Cached results retain conceptual cross-encoder and reader costs; token counts do not establish end-to-end latency.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 sara_env/bin/python -m pytest W12-POST/reranker_baseline/tests -q -p no:cacheprovider --basetemp=W12-POST/results/minilm_reranker/test_tmp
```

Source: [official model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2). The existing Week-7 wrapper uses a differently spelled repository; this independent baseline deliberately pins the exact requested official ID and revision.

# Final demo freeze status

- **Final demo version:** `evidence-lab-demo-1.0.0`.
- **Presentation comparisons:** controlled BM25/SFR candidate retrieval with common pinned SFR downstream scoring; RAW_SKIP, FIXED_SENTENCE and QBGF evidence comparisons; NONE as query-only reference; no custom correctness score.
- **Researcher-only variants:** pipeline-native comparison, exact historical fallback segmentation, and immutable saved BM25-QBGF labelled HISTORICAL NATIVE LEXICAL VARIANT. New BM25-native QBGF execution stays blocked.
- **Segmentation:** default citation/Markdown-aware, source-preserving Live adapter. No frozen splitter or benchmark preprocessing change.
- **Controlled protocol:** identical source/query, segmentation, downstream scorer, allocator, budget, serializer, Qwen reader, prompt template and decoding. RAW_SKIP uses retrieved passage order; FIXED_SENTENCE/QBGF use common semantic relevance after candidate retrieval.
- **Frozen research integrity:** 771 protected files unchanged; no benchmark, validation, confirmation or QASPER inference rerun.
- **Git/archive preservation:** source, tests, docs, dependency snapshot, checksum metadata and tokenizer assets selectively versioned on `gagan`. Curated data preserved byte-identically in versioned 13.93 MiB immutable archive; raw 98 MiB JSON, models, private/runtime data and intermediate shards excluded. Restore: `demo/.venv/bin/python demo/restore_curated_bundle.py`.
- **Tests:** 149 passed in 40.53 seconds; archive restore and controlled-comparison checks passed. An isolated Git-index checkout restored the archive and launched Curated Demo without Torch or model inference; research metadata remained read-only.
- **Launch:** `demo/.venv/bin/python -m streamlit run demo/app.py`.

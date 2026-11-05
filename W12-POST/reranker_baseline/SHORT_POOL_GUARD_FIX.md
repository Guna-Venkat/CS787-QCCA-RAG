# Frozen short-pool preflight correction

The failing QASPER query `78292bc57ee68fdb93ed45430d80acca25a9e916` has exactly eight saved SFR candidate passages. Its source, group, candidate IDs, lineage content hashes and encoder are unchanged. The new reranker guard incorrectly interpreted the configured top-10 ceiling as an exactly-ten requirement.

The guard now accepts the exact nonempty frozen pool up to ten passages. All identity/hash and four-control source checks remain strict. There is no padding, new retrieval, substitution or dropped query. Preflight lists actual short-pool widths; new reader rows retain configured `retrieval_width=10` and separately report `retrieved_passage_count`.

Full read-only preflight passed: 50 HotpotQA, 50 QASPER and 50 SciFact queries, 1,200 matched control rows. The two short pools are QASPER `78292bc57ee68fdb93ed45430d80acca25a9e916` (8) and `f20a389ace2267aa61eddcc235535452ccdae0e6` (7).

**23 tests passed.** Tests cover short frozen pools, rejection of altered membership and empty/over-width pools, plus the existing ranking, packing, leakage, resume and reader-unlock checks. The notebook's user-selected flags were preserved, including `RUN_FULL=True`; no notebook outputs were cleared.

The only changed scientific implementation file is the new baseline's `runner.py`, for provenance checks, progress metadata and actual-width reporting. Scoring/packing/reader definitions and all scientific identity fields remain unchanged. Because the identity includes source-code hashes, the previous reranker identity and nine original smoke envelopes were archived byte-for-byte under `W12-POST/results/minilm_reranker/protocol/short_pool_guard_repair_archive/`. Their payloads (3 score caches and 6 smoke selections) were preserved exactly and re-enveloped against the reviewed new identity. The repair manifest is `protocol/SHORT_POOL_GUARD_REPAIR.json`. No reader rows existed at repair time. No historical research artifacts were rewritten.

**Restart the notebook kernel**, then Run All with your existing settings, so Python imports the corrected guard. All available compatible caches resume. This repair ran unit tests and metadata/source preflight only: **no new model scoring, allocations or reader generations**.

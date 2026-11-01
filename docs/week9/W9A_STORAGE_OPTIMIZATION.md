# W9-A storage and runtime repair

BRANCH = gagan. Date: 2026-10-21. The interrupted full FIT run is preserved, not restarted from zero. No experiment, commit or push is launched by this repair.

## Why the previous run was slow and large

The grid has 19 representations × 3 retrievers × 3 widths = 171 rows per query. Every row repeated full candidate text, offsets and lineage in JSON. All three cache files also occupied memory. Dense query encoding, sparse scoring, ranking and costly evidence alignment were repeatedly performed across retrieval widths. The notebook later loaded the entire grid's candidate text again merely to show a metrics table.

## Changes

W9-A's three caches now use indexed SQLite files with lossless zlib-compressed JSON payloads. File names end in `_compact.sqlite3`; source/config/model/input identity sidecars and per-row checksums remain mandatory. Candidate text, vectors, original floating-point values, provenance and existing timings are retained exactly. No quantization, reduced grid, reduced query set or approximate evidence metric was introduced.

The cache exposes the existing key/put interface but keeps rows on disk with a bounded lookup cache. It commits every 128 rows and on normal close/interruption. An abrupt process kill can discard at most the final uncommitted CPU-only batch; SQLite rolls that batch back and resume recomputes it. Each completed batch is durable. All three files remain excluded from Git, including SQLite journals/sidecars.

Completed queries are skipped using an indexed row count. New queries are encoded once, each representation's sparse/dense scores are prepared once, each method is ranked once at the largest needed width, and smaller widths are exact prefixes. Evidence alignment is reused across methods and widths. Tests compare exact ranking dictionaries and all evidence metrics with the original calculations. New timing fields explicitly describe shared preparation/ranking, rather than claiming independently measured policy latency; saved pre-optimization timings remain unchanged. A resumed run therefore contains timings from both implementations; a publication claim about latency needs a fresh common measurement protocol, not these mixed cached timings.

Retrieval progress is printed at each unfinished query and after completion; cached queries report progress periodically. B–H load only the FIT-frozen representation/retriever/width from A's indexed database. The A notebook validates rows and input/source fingerprints but retains only scalar metrics for display. No full candidate grid is loaded into notebook memory.

## Migration and cleanup

`results/week9/patch_verification/w9a_storage_fix/compaction.json` is authoritative for status, exact row counts, source/target sizes, hashes, and deleted redundant paths. `compact_existing.py` records the one-time conversion. Before conversion it requires the exact original source hashes and unchanged model/config/input identities. It validates every old row checksum and key, rebuilds keys for the new source fingerprint, and independently reads the replacement database to check a chain hash of every scientific payload. SQLite integrity checks must also pass.

Only after all three replacements pass verification does cleanup remove the superseded large JSONL files and their identity sidecars, plus the earlier duplicate-evidence repair's obsolete local recovery copies. No checkpoints, model downloads, original datasets, W8 artifacts, tracked baseline results or completed scientific outputs are deleted. An incomplete final JSONL line, if present after interruption, is reported and discarded exactly as the old resume reader would do.

## Verification and continuation

79 W9 code tests pass, including a small complete grid/resume test, score/metric equivalence, compressed-cache identity/checksum/filter tests, abrupt-process-exit recovery and exclusive-writer locking. Another 324 ranking/metric comparisons against the original HEAD implementations match exactly (see `legacy_equivalence.json`). The notebook-default assertion is excluded because the user deliberately enabled run flags; those settings remain explicit. No full experiment was run to test this patch.

After the compaction ledger says COMPLETE, restart the W9-A notebook kernel and run it with:

```python
RUN_FULL = True
RUN_SMOKE = False
FREEZE_FIT = False
```

The existing command already has `--resume`; it will use the compact files automatically. Review complete results before freezing A. No preliminary commit is necessary. Commit the reviewed code, tests, notebook and documentation when desired, while leaving generated caches ignored.

## Completed conversion (verified)

Active caches: 10,119,879,751 bytes → 3,200,471,040 bytes. Including removal of redundant recovery copies, net space reclaimed is 9,109,417,696 bytes. Final verification measured 19,188,121,600 bytes free. Every completed payload is preserved: 24,219 chunk records, 317,520 embeddings, and 128,304 retrieval rows. These cover 750 fully completed queries and one partially completed query. Resume continues the partial query and remaining queries; it does not discard the completed 750.

`results/week9/patch_verification/w9a_storage_fix/verification.json` reports PASS. No full experiment, commit or push was launched. Before committing notebooks, restore disabled run flags as required by the safe-default test; enable RUN_FULL only for the deliberate manual experiment.

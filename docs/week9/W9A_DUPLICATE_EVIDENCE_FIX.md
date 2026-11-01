# W9-A duplicate-question evidence alignment fix

Date: 2026-10-21. BRANCH = gagan. Existing notebook edits and run settings are preserved.

The failure comes from FIT question IDs 164 and 170, paper 48. Both have question text `What dataset is used?`; QASPER contains two distinct raw question IDs for it. Their evidence sets are identical. The last printed `W9-A indexing ... question 416` line describes the indexing loop, not the later evidence lookup that fails.

`evaluation_evidence()` now accepts repeated question text only when all matching raw entries have identical evidence sets. Single-match output is unchanged. Missing matches and conflicting evidence still raise explicit errors. No arbitrary match, answer-based lookup, annotation union or gold inference feature was introduced.

All 1,746 FIT, 342 INTERNAL-VAL and 231 DEV evidence lookups pass. Four added regression cases cover unique evidence, identical duplicates in different order, conflicting duplicates and missing matches. The code suite reports 75 passed with the notebook-default check deselected. The full test invocation reports 75 passed / 1 failed because the user's saved PRE/A notebooks enable RUN_FULL/RUN_SMOKE; the test requires disabled defaults. These intentional notebook settings were not reset.

## Safe continuation

The existing source fingerprint guard would reject all pre-fix caches after editing data.py. A one-time narrowly scoped migration is stored at `results/week9/patch_verification/w9a_duplicate_evidence_fix/migrate_cache.py`, with its outcome in `migration.json`. It validates the exact prior data.py hash, all other current source/config/input/model identities, every row checksum/key, and all FIT evidence lookups. It rejects unexpected cached ambiguous-question outcomes. It changes only cache keys, fingerprints and their checksums; all chunks, embeddings, ranked candidates, metrics and timings remain unchanged.

Original cache files and identity sidecars are retained as local hard-link recovery copies under `original_cache/` before atomic replacements; those copies are ignored by Git. Check `migration.json` for COMPLETE status and exact row counts/hashes. The migration is one-time and refuses to rerun over its backups. No global stale-cache bypass or automatic compatibility exception was added. Existing smoke caches retain their old identities; they are not silently relabelled.

After COMPLETE status, restart the notebook kernel, set `RUN_FULL=True`, `RUN_SMOKE=False`, keep `FREEZE_FIT=False`, and rerun W9-A with `--resume`. The full run resumes saved work. The initial cache load/index traversal still takes time, but existing chunking, embedding and retrieval outputs are reused. Inspect the complete A results before explicitly freezing A. No full experiment was launched as part of this repair.

## Subsequent storage optimization

The later W9-A storage repair supersedes the JSONL cache format and the local recovery copies described above. Completed payloads are now preserved in verified compact SQLite caches; redundant JSONL/recovery files are removed only after full verification. See `W9A_STORAGE_OPTIMIZATION.md` and its compaction ledger for the current continuation instructions.

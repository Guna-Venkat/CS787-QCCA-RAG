# Week 11 progress

Updated: 2026-10-27T17:31:16.065624+00:00 | Branch: `gagan`

**Phase 0 is complete and reviewed. Phase A is ready for the user to start.**

This is a reviewed snapshot. Per-phase `status.json`, manifests and local query checkpoints remain the runtime records.

## Verified setup

- 150/150 real development queries: QASPER 50, HotpotQA 50, SciFact 50.
- Available sources: QASPER DEV 231; HotpotQA distractor DEV 7,405; SciFact DEV 300 plus 5,183 corpus abstracts.
- Saved code, configuration and dataset byte identities match the current inputs.
- 11,594 normalized-document/query/hierarchy checkpoint checksums verified.
- MiniLM and SFR local weight files are present; this audit did not start GPU inference.
- Setup review is recorded in `multigranular/v1/freeze_0.json`; Phase A prerequisite check passed.
- LLMLingua remains pending and does not block retrieval Phase A.
- Cleanup validation: 88 tests passed; the previously passed full historical byte rehash was not repeated.

## Phases

| Phase | State | Saved queries/results |
|---|---|---|
| 0 | COMPLETE | 150 |
| A | NOT_STARTED | 0 |
| B | NOT_STARTED | 0 |
| C | NOT_STARTED | 0 |
| D | NOT_STARTED | 0 |
| E | NOT_STARTED | 0 |
| F | NOT_STARTED | 0 |
| G | NOT_STARTED | 0 |
| H | NOT_STARTED | 0 |
| I | NOT_STARTED | 0 |

## Run the next notebook

Open `notebooks/week11/week11_A_retrieval_granularity.ipynb`. In its first FLAGS cell set:

```python
RUN_SMOKE = False
RUN_FULL = True
RESUME = True
OVERWRITE = False
```

Leave other settings unchanged, then click **Run All**. Phase A performs retrieval/embedding experiments; it does not generate answers. Prepared dataset/hierarchy caches are reused. Retrieval indexes are new work, so new indexing counters are expected.

After Phase A finishes, inspect its results before creating its review/freeze for Phase B. No outcome-based winner has been selected automatically.

## Git storage

Kept: source, configs, notebooks without generated outputs, compact real summaries, setup/progress receipts, freezes and compact setup PNGs.
Local only: synthetic smoke artifacts, cache shards, models/data, lock files, large per-unit inventories, historical byte snapshots and duplicate figure CSVs. Nothing was deleted.
No full experimental phase was launched by this cleanup/audit. Weeks 0–10 remain unchanged.

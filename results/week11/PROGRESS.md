# Week 11 progress

Updated 2026-10-27T21:57:06.628381+00:00. Branch `gagan`.

**Completed: setup 0, A, B. Running: C.** The fixed pilot remains 50 development queries per dataset.

| Phase | State | Saved logical rows |
|---|---|---|
| 0 | COMPLETE | 150 |
| A | COMPLETE | 1350 |
| B | COMPLETE | 600 |
| C | RUNNING | 0 |
| D | NOT_STARTED | 0 |
| E | NOT_STARTED | 0 |
| F | NOT_STARTED | 0 |
| G | NOT_STARTED | 0 |
| H | NOT_STARTED | 0 |
| I | NOT_STARTED | 0 |

A: 1,350 rows audited; seven QASPER mapping flags and evidence-denominator limitations recorded. B: 600 rows audited; P0 exactly reproduces A on all 150 queries. A separately recorded grouping/cache optimization matched all 403 previous B rankings exactly before 197 new rows were saved.

Full results and flags: `multigranular/v1/retrieval/INTERPRETATION.md` and `multigranular/v1/passage_representation/INTERPRETATION.md`.

C keeps SFR/P0 as the controlled primary pool. 5,550 available text-condition logical rows; 600 configured token-condition rows remain pending dependency qualification. Cached identical prompts reduce physical model calls.

The local passive monitor records every 900 seconds in `monitoring/latest.json`; the active agent reviews outcomes and launches dependent phases. Executed notebook copies are local under `executed_notebooks/`. No automatic claim of scientific superiority or fresh confirmation is made from this development pilot.

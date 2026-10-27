# Operational continuation of the frozen Week 11 pilot

The user authorized sequential full runs, delegated phase reviews and necessary
fixes, and progress checks every 15–20 minutes on 2026-10-28. This overrides the
earlier implementation-only execution restriction. Work remains on `gagan`;
Weeks 0–10, TEST and the 50-per-dataset pilot cohort remain unchanged.

## Processes and saved state

`watch_progress.py --interval 900` is a passive recorder, not a Codex scheduled
agent. It records local status/checkpoint/vector/selection counts every 15 minutes
in `results/week11/monitoring/latest.json` and a local history. It never starts
model work, chooses winners or creates reviews. The active agent performs those
steps. It warns on low disk, notebook errors, or 45 minutes with no saved progress.
The native Codex recurring-task tool was not exposed in this session.

`run_phase.py --phase C --execute` executes a full **copy** of the corresponding
notebook with smoke disabled, resume enabled, overwrite disabled and automatic
in-notebook review disabled. Originals keep their safe defaults. Copies are under
`results/week11/executed_notebooks/`; per-cell state and logs are under
`results/week11/monitoring/`. Launch dependent phases only after an audited review
and completion of the previous writer. Never run two GPU phases together.

## Current continuation

- 0: audited and frozen. A: 1,350/1,350 rows audited and frozen. B: 600/600 rows
  audited and frozen. Full A/B interpretations and flags are in each phase's
  `INTERPRETATION.md`.
- C is running. Expected: 5,550 executable text-condition logical rows. The 600
  configured token-condition rows are pending LLMLingua qualification. Identical
  evidence/prompt caching can reduce physical model calls.
- Before advancing C→D, audit per-reader baseline pairing, row checksums, all
  expected actions/queries, HELP/HURT, source/budget validation, empty evidence,
  invalid verification labels, prompt reuse, CIs and annotation limitations.
- D reuses the reviewed C textual selections. E keeps the adapted SARA checkpoint
  separate, with its own no-evidence baseline. Optional LLMLingua conditions stay
  explicitly pending; a review must explicitly list `waive_pending` to advance
  the available text-only surface without falsely declaring token work complete.
- F still requires explicit existing finalist action IDs and input dictionaries
  in the scoped `finalist_run_ids`/`finalist_configurations` fields. The user has
  delegated that review, but it must be justified and logged as exploratory.
  These late fields do not invalidate earlier freezes. Do not paste outcome
  columns into action settings or pool raw task scores across datasets.
- G is the prescribed deterministic hierarchy surface, not a trained controller.
  H/I consume saved results only. A larger-sample confirmation study needs a new
  version; do not expand the frozen pilot after seeing outcomes.

## B performance optimization and reproducibility

The reference B loop repeatedly scanned all ~100k SciFact units for every passage,
query and representation. Only the agent-owned B job was interrupted; completed
rows and original scientific source/config/manifests were preserved.

`run_phase.py --phase B --execute --cached-b` uses `fast_b.py`: the same cached SFR
vectors, same source sentences in the same order, same reference P0–P3 scoring,
same ranking/tie logic and same metrics. It groups sentence positions once per
corpus and caches ranking work. No new embedding computation is allowed by this
driver. The optimization matched **all 403 existing rankings exactly** before
saving 197 new rows. P0 rankings **and every recorded metric** match frozen A on
all 150 queries. All four pooling variants also matched the reference on synthetic
interleaved/missing-sentence tests.

Supplemental execution provenance is in B's `optimized_execution_identity.json`
and `optimization_equivalence.json`; every new row stores the driver hash. The
driver refuses a changed identity. Original launcher versions and interruption
logs are retained locally. Runtime across these epochs is not a scientific latency
comparison. Do not edit a pinned execution driver during its run; use an explicitly
recorded new epoch/version if another change becomes necessary.

The core `src/week11` and `scripts/week11` files are unchanged since the real setup
freeze. These operational launch/review tools have separately recorded hashes and
do not silently alter the scientific method.

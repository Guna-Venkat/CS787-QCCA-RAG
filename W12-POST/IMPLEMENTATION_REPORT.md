# Historical initial implementation handoff

This records the earlier unexecuted v1 delivery. The current corrected implementation and measured test/calibration status are documented in `PRE_PILOT_PATCH_REPORT.md` and `README.md`; statements below describe the historical delivery only.

# W12-POST BEA-QBGF — implementation handoff

Status: implementation delivered without executing project code, tests, smoke, preparation, generation or analysis. Only source inspection and creation of new code/notebook/metadata files took place.

## Open and run

Open `W12-POST/BEA_QBGF.ipynb`, select `sara_env`, wait until POST-W9 releases the GPU, then Run All. The default pilot automatically runs missing work, up to 900 logical outputs. Compatible finished rows resume. No manual per-cell flag changes are needed for the pilot. RUN_EXPANDED and RUN_ATTRIBUTION remain False; both need a PROMISING pilot and separate opt-in.

## New files

- `src/post_w12/bea_qbgf/protocol.py`: strict parent identity/cohorts, immutable outputs, lock, pilot/expanded metadata.
- `allocator.py`: pure bottom-up marginal selection and source escalation.
- `resources.py`: isolated writable caches, read-only old embeddings, frozen native VAL adapters and exact legacy answer verification.
- `runner.py`: automatic missing preparation, canonical controls, sequential models, exact answer reuse, durable per-row receipts/manifests.
- `analysis.py`: all-cell quality/cost/mechanism tables, 12 paired contrasts, exploratory gate, figures and report addressing ten research questions.
- `attribution.py`: optional post-pilot BEA LOO and source-group removal; exact historical control reuse where provable.
- `W12-POST/config.json`, `run.py`, `BEA_QBGF.ipynb`, `README.md`, `tests/test_bea.py`.
- New output metadata only: `AUDIT_BEFORE_RUN.md`, `PILOT_COHORT.json`, `BEA_QBGF_V1_FREEZE.json` in the isolated result root.

## Formulation choice

The explanatory second attachment supersedes the redundant novelty/need terms for v1. The algorithm uses raw global query-sentence cosine + aspect-coverage gain − max-cosine redundancy, per exact incremental token. FULL upgrades compete with every sentence action and require source accumulation. This is not a renamed FIXED_SENTENCE global ranking. Hyperparameters and gate are frozen globally before pilot outcomes; no dataset-specific tuning.

## Tests supplied, not run

Tests cover deterministic aspects, cosine space, redundancy/novelty diagnostic, coverage, source accumulation, FULL upgrade, exact costs, stopping, source order, immutable cache hits/misses, regeneration of missing answers, legacy rejection, resume without extra generation, protected frozen hashes and notebook opt-ins. The notebook offers optional execution of these and unchanged existing Week12 tests. No test-pass claim is made.

## Remaining risks

- Runtime has not been validated by execution, per instruction. First use should enable RUN_TESTS=True if desired.
- Missing caches can require all controls to be recomputed; no full-run time estimate is claimed before measured progress.
- SFR and Qwen use the frozen GPU inference implementation sequentially. No model downloads, CPU/quantized fallback or interference with POST-W9 is introduced.
- FULL utility is a sum of current sentence marginal surrogates, which can overcount correlated gain. Coverage is query-fragment cosine, not verified evidence coverage. Undefined pairwise redundancy is masked.
- The gate is exploratory and based partly on point competitiveness, not proven noninferiority. Any successful expanded result remains reused VAL, never confirmation or a final-method replacement.
- No BEA-without-expansion arm is added; attribution/activation cannot establish the isolated causal benefit of expansion.

FROZEN WEEK12 UNCHANGED
NO TESTS OR EXPERIMENTS EXECUTED

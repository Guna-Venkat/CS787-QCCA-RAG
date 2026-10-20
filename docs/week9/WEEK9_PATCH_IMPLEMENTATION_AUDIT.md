# Week 9 patch implementation audit

BRANCH = gagan. Completed 2026-10-20. Implementation and bounded checks complete; scientific experiments remain manual.

## Branch and preservation

All writes occurred on `gagan`, guarded before substantial write stages. HEAD remains `6436c6bcff2b4558b978fe478c1d34d57acc53cc`. `main` remains `fe6d1bdf08f8e9960f8a22432a98ce6fc923ea8d`. No checkout, commit, push, merge, rebase, stash, reset or clean was performed. W8 sources/results were not rewritten.

Existing W9 A–H CLI entry points, paper-fold orchestration, retrieval/chunking, selector, adaptive controller, manifest/caching architecture and TEST lock were retained. The patch extends model roles, revision mode, cost identity and reporting within that architecture. The existing A two-query retrieval smoke was retained; newly permitted work was textual PRE smoke and CPU artifact/test checks. No new full A–H experiment ran.

The missing ignored SANE source and two reference PDFs were restored from the user's local source archive and the exact cited arXiv versions. All restored reference hashes match the original `WEEK9_SOURCE_FINGERPRINTS.json`; no upstream reference implementation was edited.

## PRE and model compatibility

Added a six-official-model registry, sequential textual runtime, cache/smoke/screen CLI and notebook. Each cache checks model revision, precision, quantization, task, evidence/prompt, decoding, config, FIT input/scorer/source identity. A real cache-key defect in the existing client was fixed: the key now hashes identity values rather than iterating dictionary field names. Model-spec serialization is JSON-stable for resume.

Runtime rejects non-finite generation scores and enforces <20 GB allocated VRAM. Alternative models cannot enter SARA latent injection: both configured model and checkpoint `adapter_config.json` base-model identity are checked. A matching hidden dimension does not bypass compatibility. Canonical trained Mistral/SARA remains first pass. Optional frozen-panel extraction/revision roles use text only and unload sequentially. No model-specific projector was trained.

M0/M2/M3 caches are complete. M1/M5 access is blocked; M4 disk reserve prevents download. M0 FP16 is numerically viable with extraction format failures, M2 int8 completes the bounded interface tasks, and M3 requires NF4/FP32 to avoid invalid scores but emits no final answer within the declared budget. See `W9PRE_MODEL_AUDIT.md` for exact memory, output and scope evidence. See `W9PRE_MODEL_RESEARCH.md` and `FULL_SARA_MODEL_REPLACEMENT_FEASIBILITY.md` for official-card/literature sources and replacement limitations. No panel winner was selected.

## Revision and adaptive mode

F0/F1 prompt behavior was compared byte-for-byte with original HEAD and preserved. One predeclared evidence-anchored F2 prompt is saved with its own hash in `results/week9/revision/f2_frozen_prompt.json`. This freezes a prompt specification, not a winning revision mode. No `frozen_revision_mode.json` has been produced prematurely.

F evaluates F0/F1/F2 over identical evidence, reports all three paired differences and freezes a mode using FIT only. G loads that mode for headroom, marginal/STOP targets, replay and online final generation. Mode/prompt hashes separate caches; missing/stale freeze artifacts fail explicitly. Adaptive proposed-extraction costs include rejected proposals. Answer entailment and deployment end-to-end latency remain unmeasured rather than inferred.

## Baselines and H

The registry programmatically discovers all legitimate original notebook profiles (Standard RAG and SARA), W4 fixed BM25 summaries, actual BM25 query outputs, semantic k2/4/5/6/8, final W8 original SARA G0, HC-LoRA G0, direct HC-LoRA k4, and available G1/Uniform/OW/QCLA secondaries. Predictions are re-scored with the current canonical scorer. Summary-only rows have no fabricated per-query values or paired bootstrap.

Mandatory W8 exports absent from the gagan checkout were found in committed main history. Exactly 21 artifact files were read through Git objects and imported under W9's `baselines/imported_w8/<commit>/` archive, without changing main or copying model checkpoints. The ledger records source commit/blob/SHA256; loaders verify archived bytes and canonical summary hashes. No regeneration or baseline-name substitution occurred.

INTERNAL-VAL: 45 entries = 20 available query-level systems (342 queries each), 23 summary-only W4 entries, 2 checkpoint-only notebook profiles. DEV: 46 entries = 20 available systems (231 queries each), the same 23 summary-only and 2 checkpoint-only entries, plus 1 missing direct HC k4 entry. HC direct k4 has only INTERNAL-VAL predictions and is never relabelled as DEV. The final original SARA G0 and HC G0 predictions are available on both confirmation splits.

H retains canonical-generator architecture selection using honest outer FIT OOF outcomes. Full-FIT-trained W8 predictions are reporting-only and cannot select an OOF architecture. Main/extended/unpaired baseline tables and the generator-robustness table are distinct. F0/F1/F2 controlled ablations are reported without overriding the F-frozen mode. Robustness exports require actual frozen-panel membership, exact model revision/quantization, method and split/query/paper/scorer alignment. No cross-generator winner or robustness metrics are fabricated when predictions are absent.

## Verification

- W9: `python -m pytest tests/week9 -q`: 72 passed.
- W8 regression: `python -m pytest tests/week8 -q`: all 82 tests present on gagan passed. A test file mentioned in an older runbook is absent on this branch; it was not silently excluded from this full invocation.
- Actual M0/M2 bounded GPU smoke and resume, plus M3 bounded precision diagnostics and resume, completed with explicit PASS/FAIL records.
- PRE and F/G/H notebooks executed safely with RUN_FULL=False and RUN_SMOKE=False; no expensive experiment was triggered.
- INTERNAL-VAL/DEV baseline audits completed; archived source SHA checks and strict split guards passed.
- F0/F1 compatibility proof saved under `results/week9/patch_verification/`.
- Final source/checksum, private-output, notebook-default and diff checks are recorded in `results/week9/patch_verification/final_readiness.json`.

No FIT screen, panel selection, full W9 phase, W8 rerun, alternative projector training or TEST inference was performed. Notebook defaults remain safe; enable each experiment deliberately after its documented prerequisites/freezes. GPU smoke does not guarantee every longer input fits: runtime memory limits stay enforced on the actual screen/full workload.

## Next manual run

```bash
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage screen --model M0 --model M2 --resume
```

Then perform qualitative review and FIT panel freeze using the runbook. Continue A→H phase-by-phase; reuse any scientifically valid frozen generator-independent artifacts rather than rerunning them. Confirmation requires prior FIT freezes. TEST remains locked.

## Tracked diff summary

This excludes newly added files, which are listed below.

```text
 configs/week9/w9a.yaml                   |   2 +
 configs/week9/w9b.yaml                   |   2 +
 configs/week9/w9c.yaml                   |   2 +
 configs/week9/w9d.yaml                   |   2 +
 configs/week9/w9e.yaml                   |   2 +
 configs/week9/w9f.yaml                   |   6 ++
 configs/week9/w9g.yaml                   |   3 +
 configs/week9/w9h.yaml                   |   4 ++
 docs/week9/WEEK9_LIVE_TRACKER.md         |  17 ++++--
 docs/week9/WEEK9_RUNBOOK.md              |  73 +++++++++++++++++++---
 notebooks/week9/week9f_revision.ipynb    |  84 ++++++++++++++++++++++----
 notebooks/week9/week9g_adaptive_ke.ipynb |  84 ++++++++++++++++++++++----
 notebooks/week9/week9h_final.ipynb       |  84 ++++++++++++++++++++++----
 src/week9/backend.py                     |  34 +++++++++--
 src/week9/cost_accounting.py             |   6 +-
 src/week9/evaluation.py                  |   6 +-
 src/week9/manifests.py                   |   2 +
 src/week9/revision.py                    |  57 +++++++++++++++++-
 src/week9/runner.py                      | 100 +++++++++++++++++++------------
 tests/week9/test_week9_suite.py          |   6 +-
 20 files changed, 483 insertions(+), 93 deletions(-)
```

## Changed or added source/config/notebook/document/test files

- `configs/week9/w9a.yaml`
- `configs/week9/w9b.yaml`
- `configs/week9/w9c.yaml`
- `configs/week9/w9d.yaml`
- `configs/week9/w9e.yaml`
- `configs/week9/w9f.yaml`
- `configs/week9/w9g.yaml`
- `configs/week9/w9h.yaml`
- `configs/week9/w9pre_deepseek_nf4_fp32.yaml`
- `configs/week9/w9pre_models.yaml`
- `docs/week9/FULL_SARA_MODEL_REPLACEMENT_FEASIBILITY.md`
- `docs/week9/W9PRE_MODEL_AUDIT.md`
- `docs/week9/W9PRE_MODEL_RESEARCH.md`
- `docs/week9/WEEK9_LIVE_TRACKER.md`
- `docs/week9/WEEK9_PATCH_AUDIT.md`
- `docs/week9/WEEK9_PATCH_SOURCE_FINGERPRINTS.json`
- `docs/week9/WEEK9_RUNBOOK.md`
- `notebooks/week9/week9f_revision.ipynb`
- `notebooks/week9/week9g_adaptive_ke.ipynb`
- `notebooks/week9/week9h_final.ipynb`
- `notebooks/week9/week9pre_model_preflight.ipynb`
- `scripts/week9/audit_w9_baselines.py`
- `scripts/week9/run_w9pre_models.py`
- `src/week9/backend.py`
- `src/week9/baseline_registry.py`
- `src/week9/cost_accounting.py`
- `src/week9/evaluation.py`
- `src/week9/manifests.py`
- `src/week9/model_preflight.py`
- `src/week9/model_registry.py`
- `src/week9/model_runtime.py`
- `src/week9/revision.py`
- `src/week9/runner.py`
- `src/week9/text_model_roles.py`
- `tests/week9/test_week9_patch.py`
- `tests/week9/test_week9_suite.py`

Additional generated artifacts live under `results/week9/preflight/`, `baselines/`, `patch_verification/` and `revision/f2_frozen_prompt.json`. Raw reasoning caches and stale archives are ignored. The import ledger and patch fingerprints provide exact provenance without duplicating all predictions in this report.

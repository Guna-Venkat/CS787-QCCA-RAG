# Week 9 implementation audit

BRANCH = gagan

Git commit at audit: `29cf70a876ca8bb393af89c1cdf717903d4179cf`. No commit or push was performed. The pre-existing `.gitignore` modification was preserved. No existing source, Week-8 artifact, canonical prompt, or reference implementation was modified.

## Scope and inventory

Implemented A-H source modules, manual runners, phase configurations, safe notebooks, incremental caches, provenance, metrics, faithfulness audits, phase freezes, paper-fold orchestration, adaptive go/no-go and explicit final TEST lockout. All full experiments remain **NOT RUN**. No system is frozen and no experimental conclusion is claimed.

Added 22 source modules, 9 scripts, 8 phase configs, 8 notebooks, one test suite, reference/runbook/tracker/audit documents and source fingerprints. Results contain real two-query retrieval smoke outputs and empty reserved directories only. Early smoke caches predate final hardening and are explicitly superseded; no old results were deleted.

### Source modules

- `src/week9/__init__.py`
- `src/week9/adaptive_ke.py`
- `src/week9/backend.py`
- `src/week9/blueprints.py`
- `src/week9/caching.py`
- `src/week9/chunking.py`
- `src/week9/config.py`
- `src/week9/cost_accounting.py`
- `src/week9/data.py`
- `src/week9/evaluation.py`
- `src/week9/evidence_alignment.py`
- `src/week9/extraction.py`
- `src/week9/faithfulness.py`
- `src/week9/ke_surface.py`
- `src/week9/manifests.py`
- `src/week9/provenance.py`
- `src/week9/retrieval.py`
- `src/week9/revision.py`
- `src/week9/runner.py`
- `src/week9/sane_adapter.py`
- `src/week9/selection.py`
- `src/week9/semantic_chunking.py`

### Scripts

- `scripts/week9/run_w9_test.py`
- `scripts/week9/run_w9a_retrieval.py`
- `scripts/week9/run_w9b_ke_surface.py`
- `scripts/week9/run_w9c_selection.py`
- `scripts/week9/run_w9d_representation.py`
- `scripts/week9/run_w9e_extraction_faithfulness.py`
- `scripts/week9/run_w9f_revision.py`
- `scripts/week9/run_w9g_adaptive_ke.py`
- `scripts/week9/run_w9h_final.py`

### Notebook inventory

- `notebooks/week9/week9a_semantic_chunking_retrieval.ipynb`
- `notebooks/week9/week9b_evidence_k_surface.ipynb`
- `notebooks/week9/week9c_selector.ipynb`
- `notebooks/week9/week9d_representation.ipynb`
- `notebooks/week9/week9e_extraction_faithfulness.ipynb`
- `notebooks/week9/week9f_revision.ipynb`
- `notebooks/week9/week9g_adaptive_ke.ipynb`
- `notebooks/week9/week9h_final.ipynb`

### Configs

- `configs/week9/w9a.yaml`
- `configs/week9/w9b.yaml`
- `configs/week9/w9c.yaml`
- `configs/week9/w9d.yaml`
- `configs/week9/w9e.yaml`
- `configs/week9/w9f.yaml`
- `configs/week9/w9g.yaml`
- `configs/week9/w9h.yaml`

## References inspected

See `WEEK9_REFERENCE_AUDIT.md` for exact local paths, functions and scientific deviations. Both provided PDFs and the local SANE README, configs, prompts, helpers, retrieval/generation interfaces and evaluation code were inspected. W6-W8 routing, canonical generation, response surfaces, costs, controllers, data paths, tests and notebook/readiness artifacts were audited. Reference code under `Baselines/` is unchanged.

SemRAG is labelled **SemRAG-buffered-project-v1**: symmetric embedding buffers, adjacent cosine-distance threshold, lossless sentence ownership/final flush, explicit 1024-token cap/128-token overlap, MiniLM encoder with explicit long-input window pooling. No knowledge graph is implemented. SANE helper bodies and blueprint prompt strings are reused through a dependency-isolated AST adapter. Local SARA/Mistral/MiniLM, scientific source labels, prefix ranking and project revision prompts make this an **adapted baseline**, not a claimed exact reproduction. The final SANE baseline uses a dense candidate pool and evidence-only answer context; its concise scientific-QA answer instruction is a project adaptation.

## Scientific and provenance safeguards

- Frozen k_F=4 uses saved canonical predictions, selected FULL indices and six compressed-slot identities; W8 overlapping rows and saved greedy continuation IDs are verified.
- All 1,746 FIT, 342 INTERNAL-VAL and 231 DEV questions passed source-lineage audit. Ambiguous short substrings are resolved against the complete zero-overlap original passage partition; unresolved ambiguity still raises.
- Canonical retrieval is paper-local. Source document IDs, original passage indices, character and sentence spans survive chunking, ranking, serialization and extraction.
- Gold fields are rejected recursively by inference schemas. Evaluators run after inference. Oracles remain retrospective; marginal labels train only inside FIT folds.
- Source/config/model/prompt/input identities and per-row content checksums protect caches. Writers are locked and fsynced; unterminated last writes can be repaired, completed corrupt rows are rejected. Incompatible caches are refused, never silently overridden.
- Only complete unrestricted FIT runs freeze. H requires actual outer-heldout paper-fold outcomes, and its tolerance is applied once globally. INTERNAL-VAL is confirmation only. Ordinary FIT development rows are never relabelled OOF.
- G's global go/no-go requires verified outer FIT OOF F surfaces. Within each outer fold, its training-only architecture decision is subsequently evaluated on held-out papers. The verifier predicts proposed-action marginal utility and stops on rejection. Proposed/rejected prefix extraction costs are included.
- Every online LM prompt/generated count is separated by draft/selection/extraction/revision. Indexing is separate. Cached stage times are never presented as measured end-to-end latency.
- TEST has no generic phase option or notebook control. Its dedicated runner checks BOTH explicit arguments and a complete H FIT-OOF freeze with source/model/input/output hashes before accessing data. Default TEST data paths are absent.

## Validation actually executed

1. `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ./sara_env/bin/python -m pytest tests/week9 -q`: **56 passed**.
2. Combined Week-9 and applicable Week-8 regression run: **138 passed** (56 + 82), with the pre-existing W8-G collection failure excluded explicitly.
3. Full `pytest tests/week8 -q` was attempted and stops during collection: `tests/week8/test_w8g_repaired.py` imports absent `compute_outcome_weights` from unchanged `src/week8/hard_case_dataset.py`. This existing failure was not hidden or repaired as part of W9.
4. All eight notebooks executed through Jupyter/nbclient with `RUN_FULL=False`, `RUN_SMOKE=False`: **8 passed**, no experiments launched. Local kernel sockets required sandbox escalation; execution succeeded after it was allowed.
5. All nine CLI `--help` entry points passed. Python source and all notebook code cells compile.
6. The explicit TEST runner was called WITHOUT `--allow-test` and rejected the invocation before reading any TEST artifact. Unit tests cover the complementary missing/invalid-freeze cases. No TEST data was opened or evaluated.
7. Initial two-question real A retrieval smoke: 342 rows. Final hardened smoke: `results/week9/implementation_verification/retrieval/fit/smoke/`, with two real FIT questions, 38 representation manifests, 707 unique persisted embeddings and 342 retrieval rows. No LM answer generation. Smoke status is incomplete/nonfreezable by design. The exact smoke configuration is `results/week9/manifests/implementation_smoke.yaml`. Re-running the identical resume command left all three row counts unchanged. Every cache key remained unique; source/config identities and every row checksum matched the final implementation. No frozen JSON artifact exists.

## Fingerprints

Source bundle SHA256: `0de3eb3840bdfb59a67f4904cd5f30f9861e04bcee0fc780a9bd72c0d48718c3`.

Full source-file SHA256 values and installed dependency versions are in `WEEK9_SOURCE_FINGERPRINTS.json`. Each actual phase manifest also fingerprints real input files and model/checkpoint assets. Revalidate after any implementation/config/model change; old caches must not be silently reused.

## Known limitations and artifact gaps

- **QCLA-1 FIT-OOF export is missing.** Existing INTERNAL-VAL QCLA outcomes are present. H intentionally raises FileNotFoundError until a real paper-OOF export and training/held-out-paper sidecar are supplied. It does not substitute training scores or retune W8.
- No GPU generator call was executed during implementation. Backend integration is tested with isolated fixtures, including the repository-specific XMistral continuation contract; actual large-model runs remain manual. NVML is unavailable in the sandbox; CPU retrieval smoke succeeded.
- Full W8-G test collection has the existing missing import described above.
- Strict contiguous span support is measured; structured paraphrase entailment is unmeasured. Reference elisions/calculations can fail strict span support and remain visible for audit.
- Paragraph annotations support only coarse evidence density, not exact causal token attribution.
- Saved k4 continuation IDs are counted where available. Some legacy baseline token/overhead measurements are unavailable and explicitly null or labelled retokenized; no values are fabricated.
- Paper folds and phase freezes require substantial manual computation. No expensive experiment was launched. TEST requires explicitly configured real artifacts before final freezing and remains locked by default.

## Git checks

`git branch --show-current`: `gagan`.

`git status --short` at audit:

```text
 M .gitignore
?? configs/week9/
?? docs/
?? notebooks/week9/
?? results/week9/
?? scripts/week9/
?? src/week9/
?? tests/week9/
```

`git diff --stat` at audit (tracked files only; Week-9 additions are untracked until the user stages them):

```text
 .gitignore | 17 +++++++++++++++++
 1 file changed, 17 insertions(+)
```

The `.gitignore` diff predates this task. All implementation changes are new Week-9 files. No W8 results were overwritten. No stash, reset, clean, merge, rebase, commit or push was performed.

## Exact next action

From the repository root:

```bash
./sara_env/bin/python scripts/week9/run_w9a_retrieval.py \
  --split fit \
  --config configs/week9/w9a.yaml \
  --resume
```

For an explicitly offline session, first export `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. After completion open `notebooks/week9/week9a_semantic_chunking_retrieval.ipynb`, inspect the real results, then use the runbook's analyze-only freeze command. Do not proceed to B automatically.

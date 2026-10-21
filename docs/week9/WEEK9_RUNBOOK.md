# Week 9 manual runbook

Run from the repository root, on `gagan`. Existing W8 artifacts and `.gitignore` are preserved. No experiments run merely by opening or executing all notebook cells.

```bash
test "$(git branch --show-current)" = "gagan"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
./sara_env/bin/python -m pytest tests/week9 -q
```

FIT chooses configurations. INTERNAL-VAL is inspected development confirmation; DEV has also been inspected. Neither selects or replaces a FIT winner. TEST has no option in these phase runners or notebooks.

`k_F=4` means the original four FULL and six HEAVY passages. `retrieval_width` is W=10/20/40; `evidence_k` is K_E=0/1/2/4. F additionally measures prefix 3 so G has real adjacent-action supervision. K_E=0 returns the exact frozen draft with no additional LM calls.

## Patch execution order and model roles (2026-10-20)

BRANCH = gagan. Do not commit/push or run TEST as part of this patch. New manual order:

W9-PRE cache → W9-PRE 1–3-query smoke → W9-PRE FIT screen → freeze model panel → A retrieval → B evidence-K → C selector → D representation → E faithfulness → F F0/F1/F2 → freeze revision mode → G optional adaptive K → H method frontier plus separate generator robustness.

Reuse valid frozen A/B/C artifacts where source/config identities match. Existing smoke caches predate this patch and are preserved; they are not silently promoted or relabelled. No completed A/B/C freeze exists on this checkout. Full A results live outside the old smoke scope, so starting the documented full A run does not require deleting the old smoke.

```bash
test "$(git branch --show-current)" = "gagan"
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage cache
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage smoke --model M0 --model M2 --smoke-limit 2 --resume
```

Cache downloads only official required shards, preserves existing snapshots and an 8 GB free-disk reserve, and records inaccessible/gated/disk-limited models. It does not load models. Precision smoke tries FP16, then int8, then NF4 only on runtime/memory failure. There is one live large model at a time. Strict representative ceiling is <20,000,000,000 allocated bytes; reserved memory is also recorded. Functional empty/format/reasoning/termination flags must be inspected separately from memory/load PASS.

Only after reviewing smoke failures, manually run a screen for feasible models. Current defaults intentionally omit DeepSeek from the reading screen: NF4/FP32 fixes numerical validity but the bounded diagnostic has no final answers. M0 and Qwen2.5 are reading-screen candidates, not frozen model selections:

```bash
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage screen --model M0 --model M2 --resume
```

The FIT-only nominal 48-query subset contains whole papers and uses seeded answer-type/difficulty strata. It is fixed in `results/week9/preflight/preflight_query_ids.json`, before any model results. Original W9 chunk sentence boundaries form fixed copied-span PRE evidence when D/E is unavailable. P0 uses raw text, P1 query-only spans, P2 draft-aware spans, P3 evidence regeneration (F0), and P4 evidence-anchored draft revision (F2). Textual evidence is identical across models. T0–T4 additionally test actual model extraction and revision on 1–3 FIT queries. PRE is not a replacement for later SANE extraction experiments. Draft tokens are charged whenever a path requires the canonical first pass; rule-based PRE extraction has no LM cost.

Inspect each `screen/<model>/<precision>/qualitative_sample.json`. Raw reasoning is stored only in permission-restricted `private_raw_outputs.jsonl`, never public results. Count all reasoning tokens; missing final-answer boundaries produce an empty final answer, not a score for the reasoning. Qwen3 uses its official non-thinking switch; DeepSeek has no such switch. Deterministic DeepSeek decoding deliberately differs from its card's sampling recommendation.

Before panel freeze, create `qualitative_review.json` with `approved_model_ids` and `evidence_hash` equal to the W9 `digest()` of the parsed `textual_fit_evidence.json`. Approve only after reviewing unsupported answers. Then repeat the same screen command with `--freeze`. This reuses verified rows. Selection uses evidence-conditioned FIT F1, qualitative admissibility, one global .003 tolerance, then cumulative tokens, latency, VRAM and simpler non-reasoning behavior. At most two alternatives plus canonical Mistral are frozen. A smoke never selects a panel.

Separate roles are `first_pass_sara_model`, `extractor_model`, `revision_model`. First pass remains the existing trained Mistral/SARA checkpoint. For later textual D/E/F robustness, copy a configuration with the same FIT-frozen method and add:

```yaml
textual_model_roles:
  frozen_panel: results/week9/preflight/frozen_model_panel.json
  first_pass_sara_model: mistralai/Mistral-7B-Instruct-v0.2
  extractor_model: Qwen/Qwen2.5-7B-Instruct
  revision_model: Qwen/Qwen2.5-7B-Instruct
```

Use a separate output root for each model, with verified upstream A/B/C freezes/pools copied together without changing their content. D/E/F role settings are frozen with their protocol; confirmation cannot change them. Textual role runs receive source text/evidence only, never the SARA projector output. Canonical SARA draft cost uses its own tokenizer. Alternative extraction/revision tokenizers and character/word lengths remain explicit. Start with same-model extraction/revision; independent roles are supported through sequential unload/load, without automatically launching a Cartesian grid. H's main method run rejects alternate role configurations. Its second view consumes explicit same-split exports with `model_id,model_revision,quantization,first_pass_sara_model,model_panel_hash,frozen_method_hash,rows`; configure matching verified hashes and `generator_robustness_exports`. Rows must include unique query/paper IDs, split, current evaluation hash and actual measured costs. No robustness result is invented while exports are pending.

Baseline-only audit (no generation):

```bash
./sara_env/bin/python scripts/week9/audit_w9_baselines.py --split internal_val
./sara_env/bin/python scripts/week9/audit_w9_baselines.py --split dev
```

The final W8-H/J/G exports were recovered byte-for-byte from committed `main` history into `results/week9/baselines/imported_w8/<commit>/`, without switching or modifying branches or W8 files. Their import manifest pins Git blob IDs, commit and SHA256. Same-split original SARA/G0 and HC-LoRA/G0 are available; direct HC-LoRA k4 is INTERNAL-VAL only. Do not substitute W8-E or transfer INTERNAL-VAL rows to DEV. All original notebook entries, fixed W4 summaries and available query-level BM25/semantic outcomes are inventoried. Summary-only rows have no fabricated query IDs or paired intervals. Full-FIT trained external baselines are not FIT-OOF model-selection candidates. See `WEEK9_PATCH_IMPLEMENTATION_AUDIT.md` for actual checks and unresolved artifact availability.

## W9-A: chunking and retrieval

First full command:

```bash
./sara_env/bin/python scripts/week9/run_w9a_retrieval.py \
  --split fit --config configs/week9/w9a.yaml --resume
```

Open `notebooks/week9/week9a_semantic_chunking_retrieval.ipynb`. Inspect ORIGINAL/FIXED/SEM, BM25/SEMANTIC/RRF and W=10/20/40. Fixed size is the median semantic token length over unique FIT papers for each chunker setting; overlap is explicit. The primary selection metric is All-Gold-Evidence Recall, tolerance .003, then retrieved tokens, timed query cost where available, and complexity. MRR, nDCG and density are secondary. Annotation-free queries have null recall, not an invented success.

Freeze only after manual inspection:

```bash
./sara_env/bin/python scripts/week9/run_w9a_retrieval.py \
  --split fit --config configs/week9/w9a.yaml --resume --analyze-only --freeze
```

Outputs: `results/week9/retrieval/fit/w9a_{chunk_manifest,retrieval_results}.jsonl`, `w9a_summary.csv`, `W9A_REPORT.md`, `w9a_manifest.json`, and `results/week9/frozen/w9a_frozen_config.json`. Chunk IDs, source offsets, sentence ranges, source sections, boundary scores and original passage indices are in the chunk manifest. Embeddings are persisted separately.

This command finishes at A. It never invokes B.

## W9-B: response surface

Requires frozen A and real A outcomes on the requested split.

```bash
./sara_env/bin/python scripts/week9/run_w9b_ke_surface.py \
  --split fit --config configs/week9/w9b.yaml --resume
```

Open `notebooks/week9/week9b_evidence_k_surface.ipynb`. This uses DraftSemantic, raw evidence, frozen canonical drafts and deterministic decoding. Inspect fixed-K F1/cost, HELP/HURT, oracle and epsilon-oracle distributions in `results/week9/ke_surface/fit/`. Oracles are retrospective only.

```bash
./sara_env/bin/python scripts/week9/run_w9b_ke_surface.py \
  --split fit --config configs/week9/w9b.yaml --resume --analyze-only --freeze
```

## W9-C: selection

Requires frozen A/B and the B surface.

```bash
./sara_env/bin/python scripts/week9/run_w9c_selection.py \
  --split fit --config configs/week9/w9c.yaml --resume
```

Open `notebooks/week9/week9c_selector.ipynb`. Compare BM25, QuerySemantic, DraftSemantic, fixed-alpha QueryDraft and SANE-adapted. Alpha=.5 is preregistered, not validation-tuned. Project SANE selection requests a fixed ranking budget of four and exposes prefixes so K ablations remain nested; the separately labelled final SANE baseline requests its exact K. Synopses are query-time rule-based features; selection is an LM call. Inspect paper-bootstrap intervals against DraftSemantic, conditional useful-query metrics and cumulative costs.

```bash
./sara_env/bin/python scripts/week9/run_w9c_selection.py \
  --split fit --config configs/week9/w9c.yaml --resume --analyze-only --freeze
```

An expensive selector must exceed DraftSemantic by more than .003 F1 to enter the candidate set. Cost ordering then follows the same global tolerance.

## W9-D: representation

Requires frozen A/B/C. D0 retains y0; D1 exposes raw chunks; D2 uses the local SANE blueprint extraction prompt; D3 adds draft exposure and the confirm/complete/correct objective. Same source IDs, backend and decoding are retained.

```bash
./sara_env/bin/python scripts/week9/run_w9d_representation.py \
  --split fit --config configs/week9/w9d.yaml --resume
```

Open `notebooks/week9/week9d_representation.ipynb`. Inspect D2-D1, D3-D2 and D3-D1 paired intervals plus ALL extraction and revision token costs.

```bash
./sara_env/bin/python scripts/week9/run_w9d_representation.py \
  --split fit --config configs/week9/w9d.yaml --resume --analyze-only --freeze
```

## W9-E: structure and faithfulness

```bash
./sara_env/bin/python scripts/week9/run_w9e_extraction_faithfulness.py \
  --split fit --config configs/week9/w9e.yaml --resume
```

Requires frozen A-D. Open `notebooks/week9/week9e_extraction_faithfulness.ipynb`. E0 is a strict copied-span project format; E1 is a simple structured project format; E2 uses the actual SANE ATOMIC/COMPARATIVE/TEMPORAL prompt. Inspect `w9e_span_audit.csv` and the seeded qualitative sample. Invalid identities and unsupported quotes are flagged. E0 exposes only validated copied spans; reference tables remain unchanged for honest audit. Quote support alone does not prove fact entailment; unsupported-fact rate remains null without such measurement.

```bash
./sara_env/bin/python scripts/week9/run_w9e_extraction_faithfulness.py \
  --split fit --config configs/week9/w9e.yaml --resume --analyze-only --freeze
```

## W9-F: revision

```bash
./sara_env/bin/python scripts/week9/run_w9f_revision.py \
  --split fit --config configs/week9/w9f.yaml --resume
```

Requires frozen A-E and E extraction rows on this split. Open `notebooks/week9/week9f_revision.ipynb`. F0/F1/F2 share identical frozen evidence. F0/F1 prompts are unchanged; F2 makes source evidence primary and treats the draft as a correctable prior. Prefixes 0,1,2,3,4 allow sequential marginal supervision. The evidence at the B-frozen K is reused directly from E; other prefixes are extracted once and shared between modes. Utilization is a labelled lexical diagnostic, not causal attribution. Inspect unchanged-incomplete cases and question 1812 if present.

```bash
./sara_env/bin/python scripts/week9/run_w9f_revision.py \
  --split fit --config configs/week9/w9f.yaml --resume --analyze-only --freeze
```

## Outer paper folds: required for honest adaptive headroom and H selection

Ordinary FIT runs above are development surfaces, not OOF estimates. H refuses to freeze from them. The following is a manual, potentially expensive protocol; it is NOT executed during implementation.

For EACH `i` in 0,1,2,3,4, run A-F in order. For each phase, first run `--paper-fold i --fold-role train --freeze`, then `--paper-fold i --fold-role heldout`. Example for fold 0 A:

```bash
./sara_env/bin/python scripts/week9/run_w9a_retrieval.py \
  --split fit --config configs/week9/w9a.yaml --resume --paper-fold 0 --fold-role train --freeze
./sara_env/bin/python scripts/week9/run_w9a_retrieval.py \
  --split fit --config configs/week9/w9a.yaml --resume --paper-fold 0 --fold-role heldout
```

Repeat these two commands for B (`run_w9b_ke_surface.py`, `w9b.yaml`), C (`run_w9c_selection.py`, `w9c.yaml`), D (`run_w9d_representation.py`, `w9d.yaml`), E (`run_w9e_extraction_faithfulness.py`, `w9e.yaml`) and F (`run_w9f_revision.py`, `w9f.yaml`), retaining the fold arguments. Repeat for every fold index. Training-only phase decisions never use that fold's held-out papers. Each held-out phase requires its training-only freeze. Fold outputs live under `results/week9/folds/fold_i/`, with separate `fit` and `fit_heldout` folders. No automated all-phase launcher is provided.

## W9-G: preregistered go/no-go, then sequential verification

Requires global frozen A-F plus the five complete outer-heldout F surfaces above. Global and fold-local G validate `revision/frozen_revision_mode.json`; utility targets use that exact F0/F1/F2 generation mode.

```bash
./sara_env/bin/python scripts/week9/run_w9g_adaptive_ke.py \
  --split fit --config configs/week9/w9g.yaml --resume --collect-headroom-oof
```

Open `notebooks/week9/week9g_adaptive_ke.ipynb`. The gate proceeds only if OOF epsilon-oracle gain is at least .005 F1, or token savings are at least 15% with quality loss no more than .003. Thresholds and epsilon=.01 are in the YAML before execution. A failed gate writes `SKIPPED — insufficient adaptive headroom` and trains nothing. The oracle is a ceiling, not a deployable system.

```bash
./sara_env/bin/python scripts/week9/run_w9g_adaptive_ke.py \
  --split fit --config configs/week9/w9g.yaml --resume --collect-headroom-oof --analyze-only --freeze
```

If proceeding, candidate-specific Ridge verification uses five paper-grouped FIT folds. Features describe THIS proposed chunk/prefix; true marginal F1 is a supervised training target only. On rejection it stops immediately, avoiding unmeasured skip-prefix states. Cumulative costs include every proposed-prefix extraction, including rejected proposals, and one final answer generation. Verifier overhead in cached replay is unmeasured, not fabricated.

For each outer fold, run G training with `--paper-fold i --fold-role train --freeze`, then G heldout with `--paper-fold i --fold-role heldout`. Here its go/no-go choice is training-only inside that outer fold; H evaluates the resulting architecture out of sample.

## W9-H: final frontier

Requires all global FIT freezes plus outer A-G freezes and held-out outcomes. Core method selection retains semantic k4/k6 and W9 B5–B9 (+ B10 with complete coverage). F0/F1/F2 are also reported as controlled ablations; they do not silently override the F-frozen final mode. The baseline registry adds all available same-split historical families to reporting, with separate main, extended and unpaired/missing tables. Missing QCLA FIT-OOF is no longer a global H blocker; a full-FIT QCLA export cannot enter honest OOF selection. W8 original_SARA_G0 must come from final H/J predictions and retain its actual frozen policy/prompt provenance. Earlier E1/E5 remain legacy code adapters, never substitutes for that baseline.

For each fold:

```bash
./sara_env/bin/python scripts/week9/run_w9h_final.py \
  --split fit --config configs/week9/w9h.yaml --resume --paper-fold 0 --fold-role heldout
```

Repeat with fold indices 1-4 after their prerequisites. This evaluates the core W9 policies, reports F0/F1/F2 ablations, and includes B10 only where the fold's G passed. B10 enters the global frontier only with complete outer-fold coverage.

Collect verified OOF outcomes, apply the .003 tolerance ONCE globally, and freeze the winning architecture using its FULL-FIT frozen components:

```bash
./sara_env/bin/python scripts/week9/run_w9h_final.py \
  --split fit --config configs/week9/w9h.yaml --resume --collect-oof --freeze
```

Open `notebooks/week9/week9h_final.ipynb`. Important comparisons use 10,000 paired bootstrap resamples by paper. Report intervals; do not infer superiority from an unsupported positive point estimate. Cached stage times are not measured end-to-end latency.

## INTERNAL-VAL and DEV confirmation

Freeze each phase on FIT FIRST. Then run that phase on INTERNAL-VAL, in A→H order, by replacing `--split fit` with `--split internal_val` and omitting `--freeze`. No component is reselected. H additionally requires its global FIT-OOF final freeze before confirmation. DEV similarly uses `--split dev`; H reports available same-split registered baselines and lists missing historical predictions separately; no required row is fabricated. DEV is inspected, not blind. Gold annotations are read for evaluation only after inference.

## Explicit final TEST interface (DO NOT RUN during implementation)

Only `scripts/week9/run_w9_test.py` accepts TEST. It requires BOTH `--allow-test` and `--frozen-config results/week9/frozen/w9h_frozen_config.json`; it accepts no tuning/seed/config override flags. Before final H freezing, explicitly configure real `test_retrieval,test_reformatted,test_canonical,test_sfr,raw_test` paths in the final runtime configuration. They are deliberately absent by default. Source, model, input/output artifact hashes, FIT selection provenance and completeness are checked before accessing TEST. The exact frozen policy is executed. TEST has not been opened or evaluated during implementation.

## Resume, bounds and caches

Every phase supports `--split --config --resume --smoke-limit --seed --start-index --max-records`. Smoke rows go under a distinct `smoke/` folder and cannot freeze. Use the SAME smoke limit when resuming a smoke cache. Partial/full runs must not be treated as interchangeable: fixed-size matching in A requires the full training-paper corpus. For a new scope or changed fingerprints, use a NEW `output_root` in a copied YAML. Do not delete or silently override a stale cache. Interrupted unrestricted full runs resume naturally, including LM calls committed before a downstream interruption. JSONL rows are fsynced, checksummed and locked; an unterminated last write is repaired, while a malformed completed row is rejected.

`--analyze-only` computes summaries from validated saved outcomes and loads no generator. Keep the same collection flag on G/H analysis runs. Notebooks validate configuration and source/input fingerprints, display real saved metrics and save plots. They produce no fallback metrics when artifacts are absent.

## Known measurement limits

- Quote support is deterministic; paraphrase entailment is unmeasured.
- Evidence density is a coarse paragraph overlap proxy, not exact-token annotation.
- Long MiniLM inputs use explicit mean pooling over encoder windows; this is part of the documented project variant.
- Saved initial k4 continuation IDs are used on FIT/INTERNAL-VAL. Where original continuation IDs are absent, retokenized counts are explicitly labelled; some old W8 promotion generated counts remain null.
- W9-PRE has bounded textual GPU smoke evidence; no full W9 generation experiment has been run. This does not qualify the complete A→H SARA pipeline. Run a bounded B smoke only AFTER A is fully inspected and frozen, and A full candidate rows exist.
- All 82 W8 tests present on `gagan` pass. This branch does not contain `test_w8g_repaired.py`; no claim is made about tests from another branch.

## Additional bounded DeepSeek precision diagnostic

```bash
./sara_env/bin/python scripts/week9/run_w9pre_models.py --stage smoke --model M3 --config configs/week9/w9pre_deepseek_nf4_fp32.yaml --smoke-limit 1 --resume
```

This records `nf4_fp32` separately: NF4 weights, double quantization, FP32 compute and unquantized modules, with all generated tokens charged. It is not reported as FP16/NF4. A numerical PASS without an identifiable final answer is not interface qualification. Increasing a reasoning budget would be a new explicit FIT-only configuration with a new cache identity; no extra budget has been hidden in these results.

## W9-A compact cache and interrupted-run continuation

W9-A now stores losslessly compressed, indexed `_compact.sqlite3` caches. Resume reuses completed queries and prints retrieval progress. For the 2026-10-21 interrupted FIT run, see `W9A_STORAGE_OPTIMIZATION.md`; its verified conversion preserves all completed payloads and removes redundant JSONL/recovery files. Use `RUN_FULL=True`, `RUN_SMOKE=False`, `FREEZE_FIT=False`, restart the kernel and retain `--resume`. Review complete A results before freezing. Summaries remain CSV; large caches and SQLite journals stay ignored by Git.

## Completed A audit and current B starting point (2026-10-21)

The complete FIT A grid is audited and frozen to **FIXED-5 / SEMANTIC @20**, following the original .003 recall-tolerance then retrieved-token-cost rule. SEM-4/RRF@20 remains a useful Pareto diagnostic; it was not substituted after inspecting results. See `results/week9/retrieval/fit/freeze_audit/W9A_FREEZE_AUDIT.md` for chunk lengths, saturation, nominal median-matched controls, token-budget comparisons and the 3-D Pareto table/plot.

W9-B is ready for a manual run, with DraftSemantic fixed and K_E=[0,1,2,4]. The notebook includes both exact and epsilon=.01 oracle distributions, actual evidence counts, and a subgroup with at least four available candidates. Positive budgets currently use raw ~1,022-token chunks; this is not yet a test of one short extracted evidence unit. No B run was launched by the A audit.

```bash
./sara_env/bin/python scripts/week9/run_w9b_ke_surface.py --split fit --config configs/week9/w9b.yaml --resume
```

Use RUN_FULL=True, RUN_SMOKE=False and FREEZE_FIT=False for the deliberate notebook run. Inspect the completed surface before freezing B. Do not alter A after starting B.

## W9-B TITAN RTX OOM correction (2026-10-21)

The default BF16 SARA load used an unsupported dtype for efficient SDPA on the TITAN RTX. The fallback math attention exhausted memory at the K=4 prompt. The W9 SARA backend now explicitly loads FP16 and requires PyTorch efficient SDPA on CUDA, with no silent fallback to quadratic math attention. Checkpoint/projector weights, prompts, evidence counts and generation budgets are unchanged. This is a precision change; fresh B outputs must not mix with the original BF16 prefix.

The original small B prefix is preserved under `results/week9/patch_verification/w9b_cuda_memory_fix/archived_bf16_prefix/`. The manual B command above starts fresh FP16 caches with current source fingerprints; its `--resume` then resumes this FP16 run. Restart the notebook kernel, rerun setup, keep RUN_FULL=True, RUN_SMOKE=False, FREEZE_FIT=False and run the execution cell. A does not need rerunning.

Historical A provenance, caches and frozen configuration are byte unchanged. Because the global source bundle includes generation-only code, a narrowly pinned compatibility seal permits read-only A loading for this exact old/new source pair. AST checks confirm A execution, chunking, retrieval, scoring and identity creation are unchanged. B-H retain strict source checks; any future source change invalidates the seal. Do not rerun A under the new global source identity in its historical output directory. The proof and reproducible check are in `results/week9/patch_verification/w9b_cuda_memory_fix/a_source_compatibility.json` and `verify_a_compatibility.py`.

A bounded real FIT query 512 passed K=0/1/2/4 with the trained SARA model, canonical compressed embeddings and the full 128-token answer limit. At K=4 the prompt was 5,409 tokens, peak allocated memory 15,941,215,232 bytes (14.85 GiB), peak reserved 16,791,896,064 bytes (15.64 GiB), and the answer was nonempty. This verifies the failing first query, not every FIT prompt. Measurements: `results/week9/patch_verification/w9b_cuda_memory_fix/smoke/verification.json`. The smoke uses separate caches and does not freeze B.


## W9-C readiness after completed B (2026-10-21)

B is complete and its cost-aware primary result is frozen at K_E=0. F1 is .385631 for K0, .380967 for K1, .385004 for K2 and .380388 for K4. Do not reinterpret K2 as the primary winner. At K0, selector comparisons are identical saved drafts.

The user explicitly authorized a separate **conditional selector diagnostic at K_E=2**. Open `notebooks/week9/week9c_selector.ipynb`; it now uses `configs/week9/w9c_k2_diagnostic.yaml` and writes only C diagnostic outputs under `results/week9/diagnostics/w9c_k2/selection/fit/`. The primary config `configs/week9/w9c.yaml` is preserved. A separate B configuration in the diagnostic root pins K2, labels itself diagnostic-only and hashes the primary K0 freeze and diagnostic config. A/B source results are linked without duplication; do not run A or B under that diagnostic root.

For the deliberate full diagnostic, set RUN_FULL=True, RUN_SMOKE=False, FREEZE_FIT=False and SPLIT='fit'. Keep `--resume`. Notebook preflight verifies the diagnostic label, K2 and frozen prerequisites before loading the generator, and rejects FREEZE_FIT=True. Full and smoke outputs are separate. No automatic primary selector freeze or D-H promotion is authorized by this diagnostic; review its results first. The FIT-selected positive budget is exploratory and does not establish held-out superiority.

```bash
./sara_env/bin/python scripts/week9/run_w9c_selection.py --split fit --config configs/week9/w9c_k2_diagnostic.yaml --resume
```

Readiness evidence and the reproducible setup are in `results/week9/patch_verification/w9c_readiness/`. The B freeze was created with analyze-only; no B generations were repeated.

W9-C bounded GPU verification passed: five selector rows on FIT query 512, peak allocated memory 14.33 GiB, with DraftSemantic reproducing the corresponding saved B K2 output. Three distinct cached LM calls cover the five selector policies, including SANE selection. All 80 W9 code tests passed (the existing notebook-default check was excluded for user-enabled B flags). Final evidence: `results/week9/patch_verification/w9c_readiness/readiness.json`.

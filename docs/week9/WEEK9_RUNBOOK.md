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

Requires frozen A-E and E extraction rows on this split. Open `notebooks/week9/week9f_revision.ipynb`. F0/F1 share identical frozen evidence. Prefixes 0,1,2,3,4 allow sequential marginal supervision. The evidence at the B-frozen K is reused directly from E; other prefixes are extracted once and shared between modes. Utilization is a labelled lexical diagnostic, not causal attribution. Inspect unchanged-incomplete cases and question 1812 if present.

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

Requires global frozen A-F plus the five complete outer-heldout F surfaces above.

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

Requires all global FIT freezes plus outer A-G freezes and held-out outcomes. B0/B1 reuse canonical k4/k6; B2/B3 use real W8 E1 and the frozen selected E5 replay; B4 requires a genuine QCLA-1 paper-OOF export. No full-FIT QCLA scores are accepted as held-out results.

**Current artifact gap:** the repository contains QCLA-1 INTERNAL-VAL outcomes, but no QCLA-1 FIT-OOF export. Set `baselines.qcla_fit_oof` in all relevant YAMLs before starting H. Its JSONL rows require `question_id,paper_id,fold,prediction,prompt_tokens,cumulative_prompt_tokens,generated_tokens,generation_passes,generation_latency_sec`. The matching `.oof.json` sidecar must have `paper_grouped:true`, `training_split:"fit"` and `folds:{"0":{"training_papers":[...],"heldout_papers":[...]},...}` from real training. Fold QCLA training must exclude ALL outer-heldout papers. This implementation does not retune or overwrite W8 systems. Missing artifacts raise FileNotFoundError.

For each fold:

```bash
./sara_env/bin/python scripts/week9/run_w9h_final.py \
  --split fit --config configs/week9/w9h.yaml --resume --paper-fold 0 --fold-role heldout
```

Repeat with fold indices 1-4 after their prerequisites. This evaluates B0-B9 and B10 only where the fold's G passed. B10 enters the global frontier only with complete outer-fold coverage.

Collect verified OOF outcomes, apply the .003 tolerance ONCE globally, and freeze the winning architecture using its FULL-FIT frozen components:

```bash
./sara_env/bin/python scripts/week9/run_w9h_final.py \
  --split fit --config configs/week9/w9h.yaml --resume --collect-oof --freeze
```

Open `notebooks/week9/week9h_final.ipynb`. Important comparisons use 10,000 paired bootstrap resamples by paper. Report intervals; do not infer superiority from an unsupported positive point estimate. Cached stage times are not measured end-to-end latency.

## INTERNAL-VAL and DEV confirmation

Freeze each phase on FIT FIRST. Then run that phase on INTERNAL-VAL, in A→H order, by replacing `--split fit` with `--split internal_val` and omitting `--freeze`. No component is reselected. H additionally requires its global FIT-OOF final freeze before confirmation. DEV similarly uses `--split dev`; H requires explicit real B2/B3/QCLA DEV exports when unavailable. DEV is inspected, not blind. Gold annotations are read for evaluation only after inference.

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
- No GPU generator call has been smoke-tested in this implementation session. Run a bounded B smoke only AFTER A is fully inspected and frozen, and A full candidate rows exist.
- Full W8 test collection has a pre-existing missing `compute_outcome_weights` import in `test_w8g_repaired.py`; the other 82 tests pass.

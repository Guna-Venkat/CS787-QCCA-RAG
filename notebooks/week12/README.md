# Week 12 — QBGF development, safe execution guide

Open **week12_0_protocol_and_split_audit.ipynb**, then A, B, C, D, E, F, G, H, I. Every notebook supports Run All with its default smoke-only flags. Setup is derived before the cell that needs it.

**No full run is enabled by the implementation.** W12-C is development/validation, not confirmation. QASPER TEST is permanently blocked. Reserved Week13 confirmation candidates are never evaluated in Week12 and are not certified fresh until the historical-exposure audit is reviewed.

## What to change manually

| Notebook | Default | Manual action for real work |
|---|---|---|
| 0 | Metadata/split audit; no generation | Run All; inspect `results/week12/W12_SPLIT_AUDIT.md` |
| A | Synthetic algorithm inspection | Optional `RUN_REAL_MODEL_SMOKE=True` for the same three development queries |
| B | Synthetic checks + resources/inventory | Optional `RUN_REAL_MODEL_SMOKE=True`; `RUN_CANONICAL_SMOKE=True` qualifies one QASPER query per specialist |
| C | Synthetic development surface | Set **`RUN_FULL=True`**; effective smoke is then disabled automatically. Includes separate canonical QASPER references after qualification. |
| D | Synthetic ablations | Set **`RUN_FULL=True`** after C; no automatic winner promotion |
| E | Qualification, no installation/download/full run | Requires official package and a pinned checkpoint revision before qualification. `ALLOW_LLMLINGUA_DOWNLOAD=False` remains default. Only after qualification set `RUN_LLMLINGUA_FULL=True`. |
| F | Two-step random miniature-Qwen LoRA smoke | Prepare missing TRAIN files first, then set **`RUN_FULL_LORA_TRAIN=True`**. The notebook displays corpus counts/lengths/groups before the manually opted-in training starts. |
| G | Synthetic 2×2 schema/pipeline check | Set **`RUN_FULL=True`** only after full LoRA training completes. Real 2×2 uses NF4 for BOTH base and LoRA. |
| H | Synthetic saved-result analysis | Set `RUN_SMOKE=False` to analyze real development results; no generation |
| I | Read analysis; final method unset | Set `RUN_SMOKE=False`, then supply `FINAL_METHOD`, `FINAL_REASON`, and optional `RETAIN_LORA`. Selection is manual and checked against the preregistered conservative rule. No confirmation evaluation. |

`RESUME=True` and `OVERWRITE=False` stay enabled. A validated query/action shard is reused without generation. Preparation resumes too. Failed unfinished rows are not considered complete. A changed scientific identity requires a new configured output version; it never overwrites old results.

## Terminal equivalents

```bash
# Bounded real model smoke: THREE development queries total
sara_env/bin/python scripts/week12/run.py --phase B --real-smoke --limit 3

# Manual development runs — do not run these for smoke qualification
sara_env/bin/python scripts/week12/run.py --phase C --full --explicit
sara_env/bin/python scripts/week12/run.py --phase D --full --explicit

# Explicit data preparation, only if needed
sara_env/bin/python scripts/week12/prepare_training_data.py --download
# Or provide existing official files:
# ... --hotpot-path /path/train.json --scifact-path /path/claims_train.jsonl

# Manual TRAIN/FIT preparation and full training, isolated from SFR GPU work
sara_env/bin/python scripts/week12/run.py --phase F --full --explicit --prepare-corpus
sara_env/bin/python scripts/week12/run.py --phase F --full --explicit --train
sara_env/bin/python scripts/week12/run.py --phase G --full --explicit
```

Canonical QASPER specialists use their own Mistral checkpoints, original BM25 top10/chunk256 interface and 64-token greedy answer cap. They are side-by-side references, not matched public-Qwen competitors. SARA is k5/n10; k6/SFR variants remain unavailable until historical pool parity is established. No “QBGF beats SARA” claim is supported merely by the combined table.

## Estimated manual workload

Frozen development cohort: 50 queries each for QASPER/HotpotQA/SciFact.

- C: 2,400 logical rows, including natural top10 SFR/BM25 Qwen references; canonical Mistral references add 100 QASPER rows.
- D: 2,100 rows, including 300 QBGF-FULL rows shared with C; compatible answers reuse the generation cache.
- E: optional 300 rows if LLMLingua-2 qualifies.
- G: optional 1,200 NF4 rows for the matched LoRA 2×2.
- H/I: saved-result analysis only.
- Training cap: up to 500 TRAIN queries per dataset and six deterministic evidence/budget conditions per query, subject to actual disjoint-data availability. This is not a confirmed corpus size; F displays actual counts before training.

These are logical policy rows, not independent questions or a promised number of physical model calls. Baseline generation and compatible cache reuse affect physical counts. No serving-latency claim is made.

## Scientific safeguards

QBGF optimizes a relevance surrogate under additive token-ID block costs, including source separators. The generator consumes those exact IDs, rather than retokenizing concatenated blocks. All prospective matched text policies use this serialization. RAW_STRICT is a separately labelled historical whole-text control reproducing W11's strict-prefix allocation on the fixed W12 pool; RAW_SKIP and fixed sentences are prospective fairer controls.

Fixed sentences use skip-to-fit. C2 retains the top-eight cap and H3 its first-three-passage sentence/remaining-clause ordering; both are new-protocol ports, not direct reuse of old W11 scores. NONE and natural references keep their own explicit scope. NO_QUERY removes query-dependent allocation weights and sentence ordering inside the same query-retrieved pool; it does not remove query-dependent retrieval. NO_MASS uses uniform k/m sentence coverage while retaining query-ranked source selection. QBGF-BM25 changes retrieval AND relevance scoring and is secondary sensitivity analysis.

The 39-member primary statistical family is frozen across all three datasets and both budgets, including optional arms. Unavailable hypotheses are pending; p=1 is used internally for conservative Holm multiplicity accounting, never presented as an observed test. SciFact primary utility is macro-F1, recomputed in each cluster bootstrap; accuracy and query-level correct/incorrect HELP/HURT are separate. Strict label parsing trims whitespace only and never repairs explanations.

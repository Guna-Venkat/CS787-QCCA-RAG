# Week 12 C-EXT baseline completion

Branch: gagan. No full execution was started. Existing C was not stopped, restarted,
or edited by this patch. Its saved code identity still matches the current frozen
Week-12 implementation. Frozen protocol YAML, splits, QBGF, reader, training scheduler
and confirmation IDs are unchanged.

| Added route | Cohort / conditions | Comparison interpretation |
|---|---|---|
| BM25_RAG_RAW_SKIP_256/1024 | All frozen VAL IDs; Qwen BASE_FP16; 256/1024 | Matched whole-passage RAG using the same RAW_SKIP allocator and serialization as C |
| SFR_RAG_RAW_SKIP_256/1024 | Same VAL/budgets/prompt/parser/decoding | Exact RAW_SKIP evidence parity with C; reuse identical cached C answers |
| SARA_CANONICAL_BM25_K5_N10 | Frozen QASPER VAL only; original adapted Mistral/SARA backend | Specialist reference, not budget- or reader-matched |
| SARA_SFR_K5_N10 | Optional real one-query qualification before any full specialist run | Same specialist checkpoint, canonical 256-token splitter, prompt, 4096-dimensional BF16 latent vectors, greedy 64-token answer; SFR retrieval with canonical query instruction changes ordering |
| Historical SFR-SARA k6 | Historical W10 reference only | Never mixed into fresh Week-12 paired scores |

Budget-matched passage RAG skips oversized whole passages; it does not truncate them
or claim to present every retrieved passage. SARA uses k5 explicit/n10 total units;
its latent interface cannot honestly be described as a 256/1024-token Qwen comparison.
Optional SFR specialist deliberately fails if fewer than ten genuine source chunks
are available rather than silently changing its specified n or padding protocol.
Canonical BM25 retains its existing historical padding convention.

## Implementation and operation

- New package `src/week12_extension/` sits outside the original C code hash.
- New CLI `scripts/week12/run_c_extension.py`; default is synthetic smoke only.
- New notebook `notebooks/week12/week12_C_EXT_baseline_extension.ipynb`.
- Full work requires original C completion and manual `RUN_C_EXT_FULL=True`.
- Separate prepare/generate subprocesses release the SFR model before Qwen/Mistral loads.
- Per-query/action immutable checkpoints resume completed work. Shared C generations
  are read for exact matching evidence keys; original C results are never overwritten.
- Extension identity includes parent science/code/split/model identities and every
  extension source hash. Changed identities or checkpoints fail rather than silently reuse.
- Qualification requires a nonempty real one-query specialist answer and records the
  checkpoint hash. New specialist routes have NOT yet been GPU-qualified by this patch.
- C-EXT synthetic smoke uses three fixture queries and evaluates 12 rows (2 methods ×
  2 budgets × 3 datasets); no pretrained model is loaded.
- H/I include `merged_validation_summary.csv`, supplemental paired comparisons and
  separate specialist references. Frozen VAL/source/paper/cluster identities are checked.
- The 12 new QBGF-versus-RAG contrasts use existing paired cluster bootstrap/randomization
  conventions (10,000 draws, seed 42), with their own 12-test Holm family. These are
  supplementary exploratory contrasts added after C began, not retroactive preregistration.
  Original 39/51-test primary registries remain unchanged.
- Final I refuses to freeze until matched extension controls and canonical BM25-SARA
  QASPER reference are complete. Optional unqualified SFR-SARA stays explicitly pending.
- Large extension checkpoints stay local via gitignore; summaries/status/qualification stay visible.

## Validation

128 Week-12 unit tests passed. Includes extension budget enforcement, exact SFR/RAW_SKIP
source selection parity, repeat execution/resume, model-free smoke defaults, specialist
qualification guard and H/I integration. Notebook validation executes safe temporary
copies even when the live C notebook has full execution enabled; it never edits the
live notebook flags. All 11 notebooks passed safe-copy smoke execution, including C-EXT and the updated H/I.

No full validation, SARA reference campaign, LoRA training, confirmation or TEST was
launched. Real SARA GPU qualification is an explicit pending step, not a claimed pass.

## Manual run

After C finishes: open C-EXT, set `RUN_C_EXT_FULL=True`, Run All. Leave
`RUN_CANONICAL_BM25_SARA=True`. Set `RUN_SFR_SARA=True` only to additionally qualify and
run the optional SFR specialist. Then run H and I on saved results (`RUN_SMOKE=False`).
The CLI equivalent for matched controls is:

```
sara_env/bin/python scripts/week12/run_c_extension.py --full --explicit
```

The notebook also runs the required canonical specialist qualification/reference;
the standalone matched-control CLI does not run that specialist automatically.

Worst-case matched surface: 33,348 logical rows. With completed compatible C cache,
16,674 SFR rows reuse RAW_SKIP answers, leaving at most 16,674 new BM25 answer calls.
Canonical SARA adds 1,005 QASPER answers, plus one qualification query; optional SFR
SARA adds another 1,005 plus one qualification. These are call counts, not runtime
estimates. Empty/identical evidence can further reduce physical calls.

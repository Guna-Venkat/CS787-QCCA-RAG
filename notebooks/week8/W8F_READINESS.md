# W8-F readiness — 18 October 2026

W8-F now runs a measured, standalone semantic-k4 QCLA experiment. It compares
identity QCLA-0, query-conditioned QCLA-1, and query-plus-frozen-draft QCLA-2.
This tests adapted k4 answer generation, not an integrated W8-E controller.
Previously saved promotion outcomes must not be treated as outcomes from an
adapted generator in a future integrated experiment.

## Run

Use the `sara_env` kernel in `week8f_qcla.ipynb`. Set `RUN_FULL = True` in the first
code cell, then Run All. The notebook verifies inputs, reuses the completed real
smoke, and starts full conditioning preparation, FIT training and validation.
Leaving `RUN_FULL = False` only runs the smoke/inspection path.

The first full run encodes remaining query/draft texts with the locally cached
SFR-Embedding-Mistral model before loading SARA. Original passage embeddings are
reused. This is substantial GPU work and full training can take several hours;
it does not repeat W8-A through W8-E. At least 18 GiB free GPU memory is checked.
Checkpoints resume at completed epoch boundaries, and validation outputs resume
per query. Interrupting mid-epoch repeats that epoch on restart. Do not run two
workers against the same output folder.

Equivalent entry points:

```bash
sara_env/bin/python scripts/week8/run_w8f_qcla_experiments.py --check
sara_env/bin/python scripts/week8/run_w8f_qcla_experiments.py --smoke
sara_env/bin/python scripts/week8/run_w8f_qcla_experiments.py --full --epochs 2
```

Full results go to `results/week8/w8f_qcla/full/`, smoke results to `smoke/`, and
measured query/draft vectors to `conditioning/`. Missing inputs and changed
provenance raise errors. A completed run is verified and reused. For an explicitly
new experiment with different code/settings, choose a fresh `--output-dir`.

## Repairs

- Real 4,096-dimensional SFR query/draft embeddings replace missing zero vectors.
  Query text uses a fixed retrieval instruction; draft text is the frozen
  unadapted k4 prediction. Empty draft text is actually encoded, not zero-filled.
  SFR last-token pooling is unnormalized, with a stated 512-token text cap.
- Canonical semantic FULL/HEAVY identities and compressed-slot order are reused;
  question whitespace and prompt token counts are checked against canonical k4.
- No output LayerNorm changes the original SFR vectors. Zero-initialized residual
  output layers make both trainable variants exact identities before training.
- Missing embeddings/routing raise errors. Gold answers are target labels only.
- The exact generation prefix is retained. No extra Answer marker or BOS is
  inserted. The first gold reference supplies up to 127 content tokens plus EOS;
  generated-answer F1 uses the existing multi-reference scorer.
- Frozen SARA uses checkpointed backward passes with dropout disabled. Actual
  nonzero adapter gradients, finite gradients, updated residual weights, and
  absence of generator parameter gradients are checked.
- Gradient accumulation divides by the actual group size, including the final
  partial group. Logged training loss is the query mean of measured NLL.
- Seed 42, two fixed epochs, learning rate 1e-4, hidden dimension 2048 and
  accumulation of 16 queries are fixed before validation. Smoke uses two FIT and
  two validation queries, one epoch, and accumulation of two. No checkpoint or
  epoch selection uses the previously inspected INTERNAL-VAL split. TEST is unused.
- Trained adapters, optimizer states, source hashes, epoch histories, answer NLL,
  generated answers, F1, passage identities, costs and timings are saved.
- QCLA-2 counts both original draft and adapted generation prompts. Original
  draft latency comes from the canonical measured generation record; cumulative
  generation latency is an explicit estimate combining that record with the new
  measured pass. Batched conditioning time is separate, not labelled as deployed
  end-to-end latency.
- The two old hardcoded smoke-loss tables were removed, with a removal ledger.
  An initial real smoke is separately archived after a timing-report refinement;
  the final smoke uses the finalized source hashes.

## Validation scope

82 Week 8 tests pass. The real GPU smoke covers SFR encoding, exact canonical
baseline reproduction, frozen-SARA backward propagation, both adapters' FIT
updates, saved checkpoints, generated validation answers and source-guarded
reuse. The notebook is validated on its smoke path. Full FIT training and full
342-query confirmation remain for the user's full run; smoke F1 is not evidence
for the QCLA hypothesis. Original W8-C/W8-E results remain unchanged.

# W8-G readiness — 19 October 2026

The repaired experiment compares original SARA, matched-context uniform SFT,
hard-case LoRA and outcome-weighted LoRA. It uses complete real FIT promotion
surfaces for training and separately checks the paper-disjoint validation split.
No mock records, fabricated losses/gains or validation-selected gamma are used.

## Run

Open `week8g_outcome_weighted_sft.ipynb` with the `sara_env` kernel. Set
`RUN_FULL = True` in the first code cell and Run All. With False, the notebook
only verifies inputs and runs/reuses the real GPU smoke. Full training is left
for the user's full run; it is substantial work across three independent
trainable variants and can take many hours.

```bash
sara_env/bin/python scripts/week8/run_w8g_lora_training.py --check
sara_env/bin/python scripts/week8/run_w8g_lora_training.py --smoke
sara_env/bin/python scripts/week8/run_w8g_lora_training.py --full --epochs 2
```

Results are under `results/week8/w8g_lora/{smoke,full}/`. Source hashes guard
resumption and completed-run reuse. For an explicitly new experiment with
changed code/settings, use a fresh `--output-dir`. Only one worker may own a
folder. At least 18 GiB free GPU memory is checked before each independent model
load. Checkpoints resume at completed epoch boundaries; interrupting mid-epoch
repeats that epoch. Validation generation resumes per query/context.

## Scientific protocol

- Train only on all 1,746 FIT queries. Each query has exactly six canonical
  promotion candidates. Helpful takes priority if any delta >0.01; hurt-only
  means no helpful candidate and some delta <-0.01; all others are neutral.
- Outcome labels may construct supervised TRAINING contexts and weights:
  helpful examples expose the best FIT candidate at FULL fidelity; other cases
  retain semantic k4. Preserve original passage IDs and compressed-slot order,
  including cases where original BM25 index 0 is a compressed candidate.
- Uniform, HC and OW use identical training prompts/targets. Uniform weights are
  1. HC raw weights are 2/2/1 for helpful/hurt-only/neutral. OW raw weights are
  1+2*max_delta for helpful, 1+2*(-min_delta) for hurt-only and 1 for neutral.
  Each variant divides weights by its own full-FIT mean to control loss scale.
- Seed 42, two fixed epochs, learning rate 1e-4, rank 8, alpha 16, dropout .05,
  q/k/v/o targets, and accumulation of 16 queries are fixed before confirmation.
  No gamma/epoch/checkpoint selection uses INTERNAL-VAL. All variants are reported.
- Original SARA already has a trained PEFT adapter. New additive LoRA factors
  wrap its attention projections without merging, overwriting or nesting PEFT
  adapters. Only the new factors are trainable; base weights, original adapter
  and projector remain frozen. Every variant loads original SARA afresh. Zero
  residual identity, nonzero adapter gradients, original parameter versions,
  and absence of frozen parameter gradients are checked.
- Exact canonical generation prefix IDs are retained. The first gold reference
  is an answer-only target of up to 127 content tokens plus EOS. No second BOS
  or Answer marker is inserted, and truncation cannot erase compression slots
  or supervision. Gold answers never enter inference contexts.
- Weighted and unweighted query-mean NLL are logged separately. Accumulation
  divides by the actual group size, including the final partial group.

## Validation contexts and costs

Both contexts are confirmed on 342 INTERNAL-VAL queries:

1. `k4`: generate directly with the adapted generator on canonical semantic k4.
2. `frozen_controller`: use the existing frozen W8-E action and original k4
   draft. STOP returns that original draft. PROMOTE regenerates with the adapted
   revision generator on the actual selected passage. Original policy scores,
   candidate identities and thresholds do not change.

No true validation delta/F1/gold label selects the action. Original-SARA baselines
reuse verified real saved generations. Learned variants generate new answers;
old promotion outputs are never claimed as adapted outcomes. F1 and HELP/HURT
relative to original SARA under the same policy are reported, retaining harms.
INTERNAL-VAL is previously inspected development confirmation, not a pristine
final test. TEST is untouched.

Cumulative prompt cost includes both original draft and promoted revision passes.
New-pass timing is measured; original-draft timing is cached historical data.
Those components are labelled estimates, not measured end-to-end deployment
latency. Checkpoints save only new additive factors plus optimizer/history and
original checkpoint provenance; they are not standalone merged generators.

## Verification

93 Week 8 tests pass. Tests cover semantic identity, original SFR slot ordering,
missing/wrong-split inputs, outcome-independent validation contexts, retained
harms, exact residual identity, frozen original projections, checkpoint copying,
weighted accumulation, partial batches and target boundaries. The real smoke
uses one FIT example from each case type and two validation queries covering
STOP and PROMOTE. Its scores are correctness checks, not evidence of efficacy.

The initial partial smoke passed uniform training and generation before catching
a between-variant GPU-memory cleanup issue. It is separately archived, and the
complete smoke is repeated after fixing reference cleanup. Old synthetic smoke
tables were deleted with a removal ledger. Original W8-E/W8-F artifacts remain
unchanged. Final smoke and notebook verification are recorded under the new
W8-G result folder. All four nonempty notebook code cells executed without errors.
All 1,746 full-FIT examples were also checked in each training mode for target
bounds and exact compression-token/slot counts; the longest input is recorded in
`full_FIT_input_verification.json`. Full training itself has not been run.

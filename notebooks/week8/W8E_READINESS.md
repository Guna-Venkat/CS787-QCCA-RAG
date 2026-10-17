# W8-E repair and execution

The original token-uncertainty backfill is complete: 1,746 FIT and 342
INTERNAL-VAL queries, each checked against its canonical greedy draft. The full
pipeline completed, and W8-E plus both W8-D notebooks executed successfully.
Re-running W8-E checks source hashes and loads the frozen results; additional
SARA generation is unnecessary to inspect this run.

W8-D was closed after a targeted question-whitespace repair on 23 FIT and 5
validation masking queries. The corrected closeout found no demonstrated
attribution benefit in the tested variants, so E4 is excluded. Superseded
measurements and repair proofs are archived separately.

FIT selected E5-DraftSemantic: gate threshold 0.3, predicted HELP >0 and HURT <0.1.
Its validation F1 is 0.400080, promotion rate 42.11%, and all-query HURT rate 4.68%.
It uses 1,416.11 cumulative prompt tokens on average. This saves cost versus
always promoting, whose F1 is higher (0.412572), but does not establish a better
quality/cost operating point than static k6 (F1 0.402005; 1,352.27 prompt tokens).
The paired F1 intervals versus k4 and k6 include zero. These results do not change
the FIT-frozen selection. See `results/week8/w8e_controller/RESULTS.md` for the
full table and paper-bootstrap intervals.

## Repaired experiment

- Real canonical k4 drafts and original SARA continuation tokens/scores.
- Every decoded draft is checked against the frozen canonical answer. Selected
  token log probabilities use the actual selected IDs; float32 entropy is used.
- Padding after each answer's EOS is excluded from its measurements.
- All twelve B0 features are measured. No missing FIT token diagnostics are
  replaced with zeros or fabricated uncertainty values.
- B0-GW is fixed gain-weighted logistic regression. Sample weights are
  `1 + 5 * max(0, best_candidate_delta)`, used only for FIT training.
- Five paper-grouped FIT folds produce gate and selector OOF predictions. A
  separate full-FIT gate refit is used only for validation confirmation; the
  controller grid uses the OOF gate predictions.
- E0: return k4 draft. E1: always promote DraftSemantic. E2: B0-GW + DraftSemantic.
  E3: B0-GW + frozen U4-A. E5 variants add predicted HELP/HURT veto thresholds.
  E4 is excluded because attribution did not survive its closeout.
- Decisions read only predicted quantities, preserve original passage identity,
  and explicitly retain proposed identities when the final action is STOP.
- Retrospective outcome lookup is separate from decision-making. Harmful chosen
  promotions remain harmful outcomes; no oracle switches the output back to k4.
- Controller thresholds use actual FIT OOF downstream F1. Within 0.003 of the
  highest F1, prefer lower cumulative prompt cost, then lower cached-generation
  latency, then simpler architecture.
- FIT choices are written to `frozen_controller.json` before development
  confirmation on INTERNAL-VAL. Previous validation inspection and W8-D
  confirmation are explicitly acknowledged; this is not a pristine final TEST.
- Exact tokenizer input lengths include the chat template and special tokens.
  Every k4 input was checked token-for-token against the canonical W7 prompt.
  Promotions count both generation passes in cumulative prompt cost.
- Latency is a saved-generation estimate. Full deployed retrieval, routing, gate,
  and selector overhead has not been measured and is not fabricated as an
  end-to-end latency result.
- DAFiR-2 is not evaluated from a single-promotion surface; sequential outcomes
  would be required.

## Validation and outputs

75 Week 8 tests passed, including outcome-independent decisions, harmful-outcome
retention, candidate identity, cost ordering, selected-token alignment, and EOS
padding, canonical question normalization, and the single global quality
tolerance. All three notebooks executed without errors. The final integrity
check records source hashes, decision identities, and draft-token comparisons in
`results/week8/w8e_controller/final_verification.json`. Reproduce it with
`scripts/week8/audit_w8e_completed.py` after the notebooks have run.

The completion pipeline is `scripts/week8/run_w8e_complete.py`. It resumes
measured inputs, validates coverage and checksums, constructs the FIT controller,
confirms frozen policies, and executes the notebook with `sara_env`.

Full outputs are under `results/week8/w8e_controller/`; token diagnostics are
under `results/week8/w8e_gate_inputs/`. Completed controller runs verify their
source hashes and reuse saved results rather than silently retune on validation.
Original W8-C checkpoints and scores are unchanged. Invalid synthetic controller
outputs, including archived copies, were removed and logged in
`results/week8/w8e_controller/removed_synthetic_outputs.json`.

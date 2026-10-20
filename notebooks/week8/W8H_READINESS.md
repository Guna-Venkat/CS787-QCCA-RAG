# W8-H readiness — 20 October 2026

Code and CPU verification are complete. A real GPU smoke is still required:
this agent session has no available CUDA device and `nvidia-smi` cannot reach an
NVIDIA driver. No new full-checkpoint SARA generation or full W8-H evaluation
was run here. The notebook does not manufacture results when GPU access fails.

## Run on the GPU host

Open `week8h_full_system.ipynb` using a working GPU-enabled `sara_env` kernel.
Set `RUN_FULL = True`, leave `INCLUDE_W8G = True` to report all completed W8-G
revision adapters, and Run All. The real GPU smoke runs first; full evaluation
is blocked unless that smoke succeeds. With no GPU, the notebook only performs
preflight and displays the CPU verification, explicitly marking the GPU smoke
pending. The notebook no longer imports `src.model` before setting up its package
namespace; the runner registers the original SARA namespace before loading.

```bash
sara_env/bin/python scripts/week8/run_w8h_full_evaluation.py --check --include-w8g
sara_env/bin/python scripts/week8/run_w8h_full_evaluation.py --smoke --include-w8g
sara_env/bin/python scripts/week8/run_w8h_full_evaluation.py --full --include-w8g
```

Only DAFiR-1 is supported. DAFiR-2 needs a frozen second-state controller and
sequential training/outcomes; it is not approximated using the first-state gate.
QCLA is not automatically stacked because it changes the original draft/feature
distribution. No validation-selected checkpoint is introduced. All three W8-G
adapters are optional independent revision variants, not a selected winner.

## Repairs and inference protocol

- Regenerate original-SARA semantic k4 drafts and verify canonical text identity.
- Measure actual generated-token entropy, margins, selected-token log probability
  and EOS confidence. SARA returns continuation-only IDs: never subtract the
  prompt length. Count EOS once and exclude padding after it.
- Recompute real MiniLM semantic/MIG measurements and frozen U4-A safety
  probabilities. Load the actual FIT-trained B0-GW gate and frozen W8-E
  thresholds. No hardcoded candidate probabilities or proxy scorer remains.
- Admit only explicit pre-promotion columns to inference. Outcomes and gold
  labels are used only after the fixed action and generated answer, for scoring.
- Preserve original semantic passage IDs, FULL/HEAVY allocations and flat SFR
  slot order. Missing data raise errors instead of zero-filled embeddings.
- G0 and G1 share the exact current draft, live action and selected candidate.
  G1 inserts the draft and revision instruction inside the USER turn before chat
  formatting. STOP returns the original draft without another pass.
- Original-SARA G0 must reproduce the corresponding canonical promotion.
  Historical action agreement is diagnostic; numerical differences from live
  measured logits are recorded rather than overridden with cached decisions.
- Optional W8-G factors are loaded onto fresh original SARA for each variant.
  Existing trained adapter/projector stay intact. The shared draft/gate always
  use original SARA, not an adapted base draft.
- The current checkout had four old W8-G helper files despite completed repaired
  checkpoints. Their exact training-time versions were recovered from this
  task's earlier edits and restored only after matching the recorded SHA256
  hashes. Checkpoint/result artifacts were not changed. The restoration ledger
  lists all four exact hash matches.

## Cost and scientific scope

Exact final and cumulative prompt tokens include chat formatting and G1's added
text. Promotions count both base and revision passes; generated tokens are
counted from actual continuation IDs. Timed query-compute components cover
routing/prompt construction, SARA decoding, token diagnostics and CPU learned
inference. Retrieval/SFR are precomputed benchmark inputs; cold model loading and
deployment overhead are excluded. Branch component sums are not measured
end-to-end deployment wall times. `end_to_end_latency_sec` remains null.

This is a matched revision study with a frozen FIT-selected controller on the
previously inspected INTERNAL-VAL split. It does not tune on validation, make
TEST claims, choose the best W8-G checkpoint or establish an integrated QCLA
system. Harmful revisions remain harmful; no oracle switches back to the draft.

Results are resumable under `results/week8/w8h_full_system/{smoke,full}/`. A
completed run verifies hashes and can be inspected without CUDA. Changed
configuration/code requires a new explicitly identified experiment directory.
Only one worker may own a directory. Old proxy-score outputs and the failing
legacy notebook are archived with an invalidation ledger, not used as evidence.

## Verification completed here

- 102 Week 8 tests pass, including real tiny/random XMistral CPU generation to
  verify the continuation API. This tiny model is not a checkpoint GPU smoke.
- Real CPU gate/selector inference on all 342 validation queries using the saved
  real original-SARA token diagnostics reproduces 342/342 frozen actions and all
  144 promotions. Maximum U4-A probability difference is 2.682209e-7.
- All matched G0/G1 passage identities and exact compression-token/slot counts
  pass on those queries.
- The notebook's three nonempty code cells execute on the CPU/preflight path;
  the GPU smoke is explicitly skipped/pending, never marked passed.
- Restored W8-G support hashes match completed checkpoint provenance. Completed
  W8-E/W8-F/W8-G result artifacts remain unchanged.

See `cpu_inference_verification.json` and `readiness_verification.json` under the
new result folder. To reproduce the CPU learned-inference check, run
`sara_env/bin/python scripts/week8/verify_w8h_cpu_inference.py`.

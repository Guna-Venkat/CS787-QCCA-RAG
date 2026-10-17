# W8-C readiness audit — 17 October 2026

The notebook has been repaired and is ready to start its real-data preparation
and training workflow. The full experiment has not yet run.

1. Shut down or restart the older notebook kernel holding the SARA model. During
   this audit it occupied about 17.6 GB of the TITAN RTX, leaving only about 6 GB.
2. Open `week8c_utility_modeling.ipynb` with the registered `sara_env` kernel and
   run from the beginning. The first cell checks GPU memory before starting SARA.
3. The first run generates the missing FIT promotion surface: 1,746 queries × six
   candidates = 10,476 real generations. It reuses canonical k4 drafts and resumes
   completed promotion rows after an interruption. FIT diagnostics are not
   required: the canonical base answer in the surface is the draft.
4. Subsequent cells train U0, U1, U2-F, U2-A, lightweight joint-encoder U3, and
   learned U4-A, using five paper-grouped FIT folds and three epochs by default.
   The winner is refitted on full FIT, then evaluated on INTERNAL-VAL. Results,
   per-candidate predictions, model weights and configuration are saved under
   `results/week8/w8c_utility/`.

## Repairs

Removed synthetic FIT fallback, untrained-model evaluations, fabricated U4
fusion scores, semantic scores presented as HELP calibration, and the CLI's
INTERNAL-VAL training path. Numeric inputs now use an explicit pre-promotion
allowlist; promotion latency, outcome labels and previous model scores are
excluded. Rank 5 refers to semantic rank, not the original BM25 passage index.
MIG and draft similarity are computed from actual MiniLM embeddings. U3 budgets
all six candidate texts separately so rank 10 is retained. Saved neural models
include their trained parameters and U4 scaler; `load_selected_model` restores
these for prediction.

Metrics use each split's actual baseline and oracle values and credit all tied
best candidates. The current canonical INTERNAL-VAL surface has 342 questions,
2,052 candidates, k4 F1 **0.3890614607**, and selective-one oracle F1
**0.4806704946**. Query 739 has identical canonical generated text but a corrected
current F1 of 0.074074 instead of the historical W7 stored zero. W7 files are
unchanged; canonical text and all promotion scores are checked against current
scoring. The old W8-C comparison CSV is preserved in
`results/week8/archive_invalid_w8c_agent_run/` and must not be cited.

## Scope and verification

U1 and U2-F share the same frozen backbone, tokenizer, optimizer schedule and
seed; U2-A separately tests top-two-layer adaptation. U3 is the lightweight joint
encoder variant, not a Mistral QLoRA experiment. U4-B awaits genuine W8-D
attribution features. W8-B token uncertainty is excluded from both splits until
matching FIT diagnostics exist. Missing gold likelihood targets are masked.
Downstream F1 uses the oracle gate to isolate selection and is not an end-to-end
DAFiR controller benchmark. INTERNAL-VAL is held out from fitting, but was already
inspected during earlier development; TEST remains untouched.

Verification: 56 Week-8 tests passed, all notebook code cells parsed, real-data
semantic/MIG feature extraction passed on three INTERNAL-VAL queries, and all
five neural variants completed forward/backward/optimizer/prediction smoke steps
with cached local weights and controlled test fixtures. U3 and U4 checkpoint
round trips preserve predictions, including reordered U3 candidate inputs. Those smoke steps are
not research results. Full FIT surface generation, model OOF training and final
validation remain to be executed in the notebook.

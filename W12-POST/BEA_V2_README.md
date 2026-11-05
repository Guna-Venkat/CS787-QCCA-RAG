# BEA_V2 — manual notebook guide

Open **`W12-POST/BEA_QBGF_V2.ipynb`**, select the **sara_env** kernel, and use **Run All** after V1 has finished using the GPU. The implementation agent used CPU only and did not run the pilot.

The notebook audits historical labels, runs tests, reuses/trains CPU cross-fit logistic predictors, shows held-out diagnostics, reuses/runs allocation-only calibration on **nine queries**, verifies the freeze, prepares missing V2 selections, generates missing answers and analyzes all four systems. No manual edits between cells are needed.

| Setting | Keep |
|---|---|
| RUN_PILOT | True |
| AUTO_RUN_MISSING / RUN_MISSING | True |
| REUSE_COMPATIBLE_CACHE / RESUME | True |
| OVERWRITE / FORCE_REGENERATE | False |
| RUN_UNSEEN_EXPANSION | False |
| RUN_NEW_ATTRIBUTION | False |

**CPU versus GPU:** model training and cached-evidence allocation use CPU. The manually launched full reader stage uses the original pinned Qwen3-8B BASE_FP16 GPU reader, preserving the scientific configuration. It requires a free GPU; it must not run concurrently with V1. No automatic CPU/quantized model substitution or new model download occurs.

Current active outputs: **`results/post_w12/bea_v2/v3/`**. Earlier help-minus-harm preflight artifacts under `bea_v2/v1` and `bea_v2/v2` remain preserved; their source snapshot is in `W12-POST/archive/bea_v2_help_minus_harm_prepatch`. Neither is the historical BEA-v1 namespace, which remains `bea_qbgf/v3`.

The main pilot has 150 existing development queries × two budgets = **300 BEA_V2 logical answer rows**, before exact-prompt cache reuse. The other three methods normally reuse their existing 900 results. An interrupted run resumes immutable rows; completed compatible answers are not regenerated. Missing historical controls are restored using the canonical historical functions into V2 only. Missing/corrupt historical source or candidate problems fail loudly; they are never approximated or substituted.

```bash
# Read-only historical-data audit (writes only new V2 audit artifacts)
CUDA_VISIBLE_DEVICES='' sara_env/bin/python W12-POST/run_bea_v2.py audit
# CPU-only training and the fixed nine-query allocation calibration; no reader answers
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  sara_env/bin/python W12-POST/run_bea_v2.py smoke --execute
# Manual full development pilot; wait for V1 to finish first
sara_env/bin/python W12-POST/run_bea_v2.py pilot --execute
```

The source-budget allocator is iterative and context-dependent, **not an exact global knapsack solver**. It uses `P_help × positive semantic information gain` per exact serialized token. Source states progress through one/two/four/eight sentences, then all remaining sentences; FULL is a competing representation after two positive source contributions. Sequential additions recompute redundancy and coverage, including the context exposed by FULL. FULL may contain zero-weight sentences despite positive total transition value.

Prediction features are a fixed allowlist of source/cosine/rank/length/budget properties. Historical LOO labels are supervision only. No gold evidence, answer score/text, LOO magnitude or current-query attribution enters inference. Model training has five dataset-specific group folds; scaling is fit inside training folds. Predictors for each query exclude all examples of its query/paper/component group.

The current primary policy uses **P_help only**. P_harm and the signed help-minus-harm difference are diagnostics; neither affects evidence selection or the primary usefulness diagnostic gate. The helpful/harmful predictors, feature schema, cross-fit memberships and training hyperparameters remain unchanged.

The replacement calibration uses **three frozen queries per dataset**, both budgets, and the fixed grid `{0, .00001, .000025, .00005, .0001, .00025, .0005, .001, .0025, .005}`. It requires globally zero empty contexts, at least 20% threshold stops, at most 80% allocations using 95%+ budget, and 85% mean cosine-coverage retention relative to threshold zero. **Every dataset/budget cell** must also have mean utilization ≥20%, median evidence units ≥2, no more than one-third trivial contexts (<10% utilization or <2 units), and ≥90% mean cosine-coverage retention. These are calibration eligibility checks, not forced minimums in the allocator.

The exact protocol/sample are saved before allocation. Pick the smallest globally eligible threshold; if none qualifies, `NEEDS_REVIEW` blocks generation. No F1, gold labels/evidence or held-out LOO labels are used in allocation calibration. See `BEA_V2_HELP_ONLY_PATCH_REPORT.md` for the measured result. Stop changing the method after this freeze; the manually launched pilot supplies the next evidence, without automatic retuning.

Later unseen VAL remains disabled. Its flag currently produces a **metadata cohort plan only**, excluding trained/development groups, confirmation and any V1-expanded groups. It does not run unseen inference: all-development predictors and new canonical source preparation need a separately reviewed extension after the complete pilot decision. Cross-fit predictors are never silently applied to unseen queries. New attribution is outside this version.

See `BEA_V2_HELP_ONLY_PATCH_REPORT.md` for the current amendment and calibration; `BEA_V2_IMPLEMENTATION_REPORT.md` records the earlier help-minus-harm implementation. No current V2 answer-quality results exist until you launch the notebook.

# W12-QBGF-v3 scheduler, CORE_FINAL and LLMLingua qualification patch

Frozen W12-QBGF-v3 YAML, all split IDs/hashes, QBGF source/definitions and historical artifacts are unchanged. No split was regenerated. Split hash: `78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117`. This user-authorized pre-full-run patch supersedes replacement-sampling instructions; original audit artifacts remain preserved.

## Exact coverage and loss weighting

| dataset   |   eligible |   planned_unique |   actual_unique |   actual_repeats |   loss_weight |   nominal_epoch_loss_mass |   planned_update_touches |   planned_update_loss_coefficient_mass |   actual_applied_updates |
|:----------|-----------:|-----------------:|----------------:|-----------------:|--------------:|--------------------------:|-------------------------:|---------------------------------------:|-------------------------:|
| HotpotQA  |      90447 |            90447 |               0 |                0 |      0.343317 |                     31052 |                     5823 |                                1941.01 |                        0 |
| QASPER    |       2178 |             2178 |               0 |                0 |     14.2571   |                     31052 |                     1842 |                                1940.75 |                        0 |
| SciFact   |        531 |              531 |               0 |                0 |     58.4783   |                     31052 |                      509 |                                1940.75 |                        0 |

One seeded permutation of canonical `(dataset,query_id)` identities, without replacement. Every one of93,156 eligible TRAIN examples occurs exactly once in the registered epoch;5,823 optimizer updates at batch1/accumulation16, including a final four-example update. `w_d=N/(3*N_d)` multiplies query-mean answer NLL. Nominal epoch loss mass is31,052 per dataset. A separate balanced replacement sampler is smoke-only. No evidence mode, target, split, LoRA hyperparameter or validation objective changed.

**Actual final-training consumption is zero for every dataset.** Planned values are not observed training. The toy CPU gradient/ledger test consumes synthetic identities only. During manual full training the durable forward/update ledger reports physical unique examples, repeats across restarted attempts, unapplied/skipped windows, applied weights and optimizer-step contributions. Step touches overlap for mixed-dataset accumulation. Coefficients use the actual window denominator, including the short last window; clipping and Adam prevent interpreting them as gradient norms or equal parameter changes. Completion refuses missing TRAIN examples. Interrupted work is retained in the physical-consumption audit rather than falsely reported as zero repeats.

## C/D cost and execution

| phase   | CORE_FINAL   |   queries |   logical_rows |   additional_call_upper_bound |
|:--------|:-------------|----------:|---------------:|------------------------------:|
| C       | True         |      8337 |          91707 |                         83370 |
| C       | False        |      8337 |         133392 |                        125055 |
| D       | True         |      8337 |          33348 |                         16674 |
| D       | False        |      8337 |         116718 |                        100044 |

CORE C: NONE, RAW_SKIP, FIXED_SENTENCE, H3, QBGF at256/1024, plus C2 at256 only. CORE D: QBGF and NO_QUERY at256/1024. This covers the existing C/D preregistered primary contrasts. The unchanged39-member family retains separately opted-in E/G contrasts as pending; no hypotheses are silently removed to improve multiplicity correction.

Core C/D total125,055 logical rows (some repeated NONE/QBGF rows) and at most100,044 additional answer calls assuming compatible C-generation reuse in D. Broader portable C/D would require250,110 logical rows and up to225,099 calls, plus2,010 QASPER specialist calls if enabled. Identical/empty evidence can reduce calls further. These are bounds, not measured full-run times. At generation caps, the core upper bound is12,567,744 generated tokens across QA/SciFact; real answer lengths and retrieval/embedding work remain unknown. No hour estimate is inferred from tiny smoke.

All full flags remainFalse. `CORE_FINAL=True` is now present in notebooks. For C/D manually enable RUN_FULL only after review. Secondary arms require CORE_FINAL=False or CLI `--include-secondary`; they use separate result directories and are not automatically launched. Natural-budget RAG, RAW_STRICT, nonprimary C2@1024 and the other QBGF sensitivities remain implemented. A/B remain qualification/smoke, E/G separate optional manual experiments. Primary budgets and scientific definitions are untouched.

## LLMLingua-2

Official checkpoint metadata was checked before download: smaller mBERT709,388,104 bytes(0.661 GiB), versus larger XLM-R2,235,829,648 bytes(2.082 GiB). Measured free space before download was17.10 GiB; qualification retained16.41 GiB, exceeding the10 GiB reserve. No caches were deleted. Dependencies were installed in a separate environment sharing the original pinned torch/transformers read-only; the original environment was not upgraded.

**QUALIFIED:** `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` @ `5f0c82792b7ea14c6484e015b6a072009496b7f2`; LLMLingua0.2.2. Official Microsoft implementation: https://github.com/microsoft/LLMLingua and https://huggingface.co/microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank. Synthetic source2,041 Qwen tokens compressed to250 at budget256 and875 at budget1024 using the predefined retries. These are mechanical budget checks, not quality evidence; no answer LM was called during compressor qualification. The original frozen YAML remains intact; a checksummed optional baseline runtime lock records model, revision, exact weight hash, package versions and worker identity. The final-method freeze captures the effective compressor spec. It does not silently launch E or add compression to QBGF.

Remaining disk is not a guarantee that all-TRAIN embedding preparation/checkpoint storage will fit. Keep E optional and arrange capacity before large user-run workloads.

## Verification and manual use

120 tests pass;10/10 default notebooks pass; three frozen VAL-query execution smokes pass source/budget checks. The synthetic weighted-gradient ledger test verifies full coverage, zero planned repeats, actual update accounting and restart repeats. Frozen role/TEST guards still pass. All2,721 protected tracked files/config/QBGF checksums were verified, allowing only the requested non-scientific ignore-file update; no historical result artifact changed.

Manual: notebook0→A→B→C(core)→D(core)→optional E/F/G→H→I. F still requires RUN_TINY_LORA_SMOKE=False and RUN_FULL_LORA_TRAIN=True. `sampling_plan.json` is planned; `sampling_actual.json` is observed. Week13 remains locked until final freeze. No full validation, full LoRA training, confirmation or QASPER TEST was run.

# Week 12 Part I — final method freeze closeout

**Status: COMPLETE.** The final method is frozen as `FIXED_SENTENCE` with SFR retrieval and the pinned Qwen BASE_FP16 reader, at both256/1024 evidence-token budgets. No LoRA is retained. No confirmation experiment was started.

Freeze time: **01 November 2026, 14:57:41 IST**. Branch: `gagan`. Source implementation commit captured in the freeze: `93992932e29252bcedbc56bda02e0c8ae3926001`.


## 1. Final system

| component                   | frozen value                                                    |
|:----------------------------|:----------------------------------------------------------------|
| Method                      | FIXED_SENTENCE                                                  |
| Retriever                   | SFR: Salesforce/SFR-Embedding-Mistral                           |
| Retrieval width             | 10                                                              |
| Reader                      | Qwen/Qwen3-8B                                                   |
| Precision                   | BASE_FP16                                                       |
| Evidence budgets            | 256 and1024; excludes query/instruction framing                 |
| Adaptation                  | None; retain_lora=false                                         |
| Tokenizer / prompt / parser | Existing pinned Week12 Qwen tokenizer and reader implementation |
| Data used for selection     | Frozen VALIDATION only                                          |

The final policy remains a single sentence-selection policy across datasets, not six post-hoc per-cell winners. Both primary budgets are retained for controlled confirmation; I did not pick a new adaptive budget rule. No retriever, split, QBGF definition or historical outcome was changed.


## 2. Recorded rationale

> Retain the preregistered fixed-sentence fallback: QBGF does not establish noninferiority across all dataset/budget cells; mixed-granularity benefits are unproven. E/F/G were not run; retain no LoRA.

The method is the **preregistered simple fallback**, not a claim that fixed sentences maximize F1 in every cell. QBGF does not establish noninferiority across all required dataset/budget comparisons. Its small token savings do not rescue the failed quality gate. Query-conditioned allocation showed value in D, while a consistent advantage from allowing mixed granularity was not established.


## 3. Final policy VAL results

| dataset   |   budget |   n_queries |   quality |     HELP |     HURT |   answer_changed |   evidence_tokens |   prompt_tokens |   generated_tokens |
|:----------|---------:|------------:|----------:|---------:|---------:|-----------------:|------------------:|----------------:|-------------------:|
| HotpotQA  |     1024 |        7155 |  0.423310 | 0.590217 | 0.089727 |         0.870860 |       1001.333054 |     1055.249057 |          14.664151 |
| HotpotQA  |      256 |        7155 |  0.408247 | 0.516841 | 0.094200 |         0.869322 |        254.690566 |      308.606569 |          13.708176 |
| QASPER    |     1024 |        1005 |  0.298788 | 0.627861 | 0.068657 |         0.977114 |       1023.005970 |     1065.683582 |          20.313433 |
| QASPER    |      256 |        1005 |  0.263430 | 0.551244 | 0.100498 |         0.983085 |        255.215920 |      297.893532 |          19.082587 |
| SciFact   |     1024 |         177 |  0.670398 | 0.395480 | 0.141243 |         0.598870 |       1021.045198 |     1085.983051 |           3.338983 |
| SciFact   |      256 |         177 |  0.700062 | 0.389831 | 0.124294 |         0.542373 |        253.310734 |      318.248588 |           3.451977 |

Quality is mean answer F1 for QASPER/HotpotQA and pooled three-class macro-F1 for SciFact. HELP/HURT is relative to the matched no-evidence control under the original task convention. These are already observed development estimates, not fresh confirmation results. No cross-dataset metric average is used.


## 4. Why QBGF was not promoted

| dataset   |   budget |     point |    ci_low |   ci_high |   NI_margin | NI_established   |   joint_holm_adjusted_p |
|:----------|---------:|----------:|----------:|----------:|------------:|:-----------------|------------------------:|
| QASPER    |      256 | -0.027915 | -0.040996 | -0.014908 |    0.003000 | False            |                0.005099 |
| QASPER    |     1024 | -0.003734 | -0.012490 |  0.005008 |    0.003000 | False            |                1.000000 |
| HotpotQA  |      256 |  0.003821 | -0.003511 |  0.011013 |    0.003000 | False            |                1.000000 |
| HotpotQA  |     1024 |  0.009033 |  0.004544 |  0.013509 |    0.003000 | True             |                0.022198 |
| SciFact   |      256 | -0.025094 | -0.087757 |  0.036542 |    0.010000 | False            |                1.000000 |
| SciFact   |     1024 | -0.009985 | -0.071143 |  0.050456 |    0.010000 | False            |                1.000000 |

Difference is QBGF minus fixed sentences. Nominal95% CIs use10,000 paired paper/query/document-group resamples, seed42. The existing gate requires every CI lower bound to clear its dataset-specific noninferiority margin. Only HotpotQA1024 passes; five cells fail to establish noninferiority. Failure to establish noninferiority is not proof of harm in every such cell. Original and joint multiplicity corrections remain recorded; missing E/F/G outcomes are never imputed as zeros.


## 5. Frozen confirmation controls

| method         | reader    | budgets    | role                  |
|:---------------|:----------|:-----------|:----------------------|
| NONE           | BASE_FP16 | 256 / 1024 | prevalidated control  |
| RAW_SKIP       | BASE_FP16 | 256 / 1024 | prevalidated control  |
| FIXED_SENTENCE | BASE_FP16 | 256 / 1024 | PRIMARY FROZEN POLICY |
| H3             | BASE_FP16 | 256 / 1024 | prevalidated control  |
| QBGF_FULL      | BASE_FP16 | 256 / 1024 | prevalidated control  |

Only existing controls with complete BASE_FP16 VAL coverage at both budgets were retained. This list is fixed before any confirmation generation. QBGF remains a comparison system, even though it was not promoted to the final primary policy. Reusing the freeze does not authorize method changes after confirmation outcomes are seen.


## 6. Explicit omissions

| condition                                 | reason                                                                             |
|:------------------------------------------|:-----------------------------------------------------------------------------------|
| RAW_STRICT                                | Unrun secondary VAL condition                                                      |
| C2                                        | Evaluated at256 only; not validated at1024 for the uniform two-budget runner       |
| STRONG_SFR_RAG                            | Unrun unrestricted secondary VAL condition                                         |
| BM25_QWEN_REFERENCE                       | Unrun unrestricted secondary VAL condition                                         |
| LLMLingua-2 / E                           | Qualified runtime only; no VAL quality evaluation; excluded from confirmation      |
| LoRA / F and G                            | Full training and paired validation were skipped; no adapter retained              |
| D single-granularity / NO_QUERY ablations | Diagnostic ablations, not added to the original uniform MAIN confirmation registry |
| C-EXT SARA / historical k6                | Specialist/historical references; not matched BASE_FP16 confirmation controls      |

Full E/F/G were omitted by user decision because of runtime cost. Qualification/synthetic smoke is not a replacement quality result. C-EXT budget-matched BM25 RAG remains in validation reports; it is not silently equated with the unrestricted BM25_QWEN_REFERENCE control. Historical SARA references remain separate and no fresh QASPER confirmation is available.


## 7. Reserved confirmation and workload

| dataset   |   queries |   groups | status                          |
|:----------|----------:|---------:|:--------------------------------|
| HotpotQA  |       200 |      200 | reserved, not evaluated         |
| QASPER    |         0 |        0 | no fresh confirmation available |
| SciFact   |        50 |       43 | reserved, not evaluated         |

The cohort is unchanged: **250 queries total**, HotpotQA200 and SciFact50 (43 SciFact groups). QASPER contributes zero fresh confirmation queries. Historical QASPER TEST remains closed. Only membership metadata was inspected for this report; confirmation examples/labels were not loaded for reader evaluation.

The frozen five-method × two-budget grid has **2,500 logical result rows**. With identical NONE answers shared across budget labels, the conservative generation upper bound is **2,250 distinct answer calls** before other identical-evidence cache reuse. Retrieval/embedding preparation is additional work; no runtime estimate is inferred from call counts. None of these calls were started by this closeout.


## 8. Verification and immutable artifacts

| check                          | result                                                                        |
|:-------------------------------|:------------------------------------------------------------------------------|
| Two freeze artifacts           | Byte-identical                                                                |
| Original scientific identity   | Matches current frozen configuration, pinned models/packages and source files |
| Frozen confirmation membership | Matches original split metadata                                               |
| Reviewed freeze helper         | Source SHA256 matches recorded review_source_hash                             |
| H evidence tables              | All three saved table hashes match                                            |
| Primary policy / adapter       | FIXED_SENTENCE / no LoRA                                                      |
| Confirmation controls          | Exactly the five validated BASE_FP16 controls                                 |
| Week13 execution               | No results/week13 directory; no evaluation started                            |

Artifacts:

- `results/week12/W13_FINAL_METHOD_FREEZE.json`
- `results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json`

Both have SHA256 `d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b`. Split hash: `78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117`.

The freeze records protocol, pins, reader precision, parser/prompt hashes, method list, optional-phase decisions, H table hashes, selection evidence and exact confirmation IDs. Re-running I with an identical reviewed decision reuses the existing timestamped freeze; conflicting decisions fail rather than overwrite. Report creation did not regenerate or edit either artifact.


## 9. Next steps and limits

Part I is closed. The separate150-VAL-query attribution J/K/L diagnostic is implemented and may be run manually; it is not a new tuning set and must not change confirmation IDs or redefine primary claims. The full attribution study was not launched by this report.

When separately authorized, Week13 can validate readiness using:

```bash
sara_env/bin/python scripts/week13/run_confirmation.py
```

Without full/explicit flags this performs metadata readiness checks only. Actual confirmation remains a separate manual action, not an effect of executing I or this report.

Use confirmation only to evaluate the frozen system and controls. QASPER has no fresh generalization claim, SciFact confidence intervals may be wide, and no LLMLingua/LoRA result may be claimed. These restrictions remain true after a successful freeze.


## 10. Closeout checks

The actual saved freeze passed the Week13 loader's identity/membership validation plus the byte-equality, reviewed-source and H-table-hash checks above. The reviewed freeze code previously passed139 Week12 tests; the targeted final-review guards were rerun during this closeout. No new models, training, attribution generations or confirmation evaluation were run.

# Week 12 H — final validation interpretation and Part I recommendation

Date: 1 November 2026. Branch: `gagan`. Saved C/C-EXT/D only; no new model generations.

**Recommendation: freeze `FIXED_SENTENCE` with `RETAIN_LORA=False` under the existing preregistered fallback rule.** This is the defensible default when the proposed system does not pass the joint quality/cost promotion gates; it is not a claim that fixed sentences maximize F1 on every dataset/budget.


## 1. What H establishes

C completed91,707 logical rows and D66,696; their repeated FULL rows deduplicate to **141,729 rows / 51 conditions** in H core analysis. C-EXT adds33,348 rows /12 method-budget labels; its SFR RAW_SKIP labels duplicate original RAW_SKIP. All displayed matched systems use the pinned BASE_FP16 Qwen reader at256/1024. Two QASPER SARA references use adapted Mistral separately.

QASPER1,005 queries /281 papers; HotpotQA7,155 /7,155 query groups; SciFact177 /157 document/claim groups. Metrics are QA answer F1 and SciFact three-class macro-F1. No pooled cross-dataset score is constructed. These are development/VAL estimates, not fresh confirmation results.

E/F/G are explicitly **NOT_RUN_BY_USER_DECISION**. LLMLingua qualification is not a quality evaluation; no compressor or LoRA quality score is imputed. Original39-test and joint51-test families remain unchanged.


## 2. QASPER — complete matched quality/cost summary

| method             |   budget |   n_queries |   quality |   evidence_tokens |   prompt_tokens |
|:-------------------|---------:|------------:|----------:|------------------:|----------------:|
| FIXED_SENTENCE     |      256 |        1005 |  0.263430 |        255.215920 |      297.893532 |
| C2                 |      256 |        1005 |  0.258118 |        227.702488 |      270.380100 |
| H3                 |      256 |        1005 |  0.254363 |        235.041791 |      277.719403 |
| QBGF_SENTENCE_ONLY |      256 |        1005 |  0.235584 |        252.600000 |      295.277612 |
| QBGF_FULL          |      256 |        1005 |  0.235515 |        252.675622 |      295.353234 |
| QBGF_PASSAGE_ONLY  |      256 |        1005 |  0.169283 |        224.719403 |      267.397015 |
| RAW_SKIP           |      256 |        1005 |  0.169283 |        224.719403 |      267.397015 |
| BM25_RAG_RAW_SKIP  |      256 |        1005 |  0.153388 |        237.789055 |      280.466667 |
| QBGF_NO_QUERY      |      256 |        1005 |  0.147303 |        252.561194 |      295.238806 |
| NONE               |      256 |        1005 |  0.082157 |          0.000000 |       42.677612 |
| H3                 |     1024 |        1005 |  0.299487 |       1003.603980 |     1046.281592 |
| FIXED_SENTENCE     |     1024 |        1005 |  0.298788 |       1023.005970 |     1065.683582 |
| QBGF_SENTENCE_ONLY |     1024 |        1005 |  0.297740 |       1019.838806 |     1062.516418 |
| QBGF_FULL          |     1024 |        1005 |  0.295055 |       1020.468657 |     1063.146269 |
| QBGF_PASSAGE_ONLY  |     1024 |        1005 |  0.289581 |        936.052736 |      978.730348 |
| RAW_SKIP           |     1024 |        1005 |  0.287957 |        912.558209 |      955.235821 |
| BM25_RAG_RAW_SKIP  |     1024 |        1005 |  0.255928 |        908.189055 |      950.866667 |
| QBGF_NO_QUERY      |     1024 |        1005 |  0.228373 |       1019.801990 |     1062.479602 |
| NONE               |     1024 |        1005 |  0.082157 |          0.000000 |       42.677612 |

Budget is selected-evidence tokens; final prompt cost includes framing/query. RAW_SKIP denotes SFR budget-matched whole-passage RAG; BM25_RAG_RAW_SKIP is the matched lexical retrieval control. The duplicate SFR extension alias is omitted here, not treated as independent evidence.


## 2. HotpotQA — complete matched quality/cost summary

| method             |   budget |   n_queries |   quality |   evidence_tokens |   prompt_tokens |
|:-------------------|---------:|------------:|----------:|------------------:|----------------:|
| H3                 |      256 |        7155 |  0.425311 |        237.482600 |      291.398602 |
| QBGF_SENTENCE_ONLY |      256 |        7155 |  0.412189 |        250.840671 |      304.756674 |
| QBGF_FULL          |      256 |        7155 |  0.412068 |        250.842208 |      304.758211 |
| C2                 |      256 |        7155 |  0.411927 |        220.700489 |      274.616492 |
| FIXED_SENTENCE     |      256 |        7155 |  0.408247 |        254.690566 |      308.606569 |
| RAW_SKIP           |      256 |        7155 |  0.406974 |        224.144934 |      278.060936 |
| QBGF_PASSAGE_ONLY  |      256 |        7155 |  0.393509 |        228.751223 |      282.667226 |
| BM25_RAG_RAW_SKIP  |      256 |        7155 |  0.379553 |        225.550804 |      279.466806 |
| QBGF_NO_QUERY      |      256 |        7155 |  0.308593 |        250.986164 |      304.902166 |
| NONE               |      256 |        7155 |  0.165324 |          0.000000 |       53.916003 |
| QBGF_PASSAGE_ONLY  |     1024 |        7155 |  0.452507 |        956.906639 |     1010.822642 |
| RAW_SKIP           |     1024 |        7155 |  0.452133 |        951.751363 |     1005.667365 |
| H3                 |     1024 |        7155 |  0.436421 |        987.392872 |     1041.308875 |
| BM25_RAG_RAW_SKIP  |     1024 |        7155 |  0.435345 |        959.903704 |     1013.819706 |
| QBGF_FULL          |     1024 |        7155 |  0.432344 |        993.162124 |     1047.078127 |
| QBGF_SENTENCE_ONLY |     1024 |        7155 |  0.432295 |        992.543117 |     1046.459119 |
| FIXED_SENTENCE     |     1024 |        7155 |  0.423310 |       1001.333054 |     1055.249057 |
| QBGF_NO_QUERY      |     1024 |        7155 |  0.412252 |        990.339623 |     1044.255625 |
| NONE               |     1024 |        7155 |  0.165324 |          0.000000 |       53.916003 |

Budget is selected-evidence tokens; final prompt cost includes framing/query. RAW_SKIP denotes SFR budget-matched whole-passage RAG; BM25_RAG_RAW_SKIP is the matched lexical retrieval control. The duplicate SFR extension alias is omitted here, not treated as independent evidence.


## 2. SciFact — complete matched quality/cost summary

| method             |   budget |   n_queries |   quality |   evidence_tokens |   prompt_tokens |
|:-------------------|---------:|------------:|----------:|------------------:|----------------:|
| C2                 |      256 |         177 |  0.710248 |        229.853107 |      294.790960 |
| FIXED_SENTENCE     |      256 |         177 |  0.700062 |        253.310734 |      318.248588 |
| H3                 |      256 |         177 |  0.685627 |        230.694915 |      295.632768 |
| QBGF_FULL          |      256 |         177 |  0.674969 |        250.604520 |      315.542373 |
| QBGF_SENTENCE_ONLY |      256 |         177 |  0.674969 |        250.604520 |      315.542373 |
| QBGF_NO_QUERY      |      256 |         177 |  0.487967 |        252.819209 |      317.757062 |
| QBGF_PASSAGE_ONLY  |      256 |         177 |  0.457775 |        186.932203 |      251.870056 |
| RAW_SKIP           |      256 |         177 |  0.456378 |        186.135593 |      251.073446 |
| BM25_RAG_RAW_SKIP  |      256 |         177 |  0.437511 |        184.033898 |      248.971751 |
| NONE               |      256 |         177 |  0.420662 |          0.000000 |       64.937853 |
| BM25_RAG_RAW_SKIP  |     1024 |         177 |  0.710716 |        938.841808 |     1003.779661 |
| RAW_SKIP           |     1024 |         177 |  0.698979 |        934.000000 |      998.937853 |
| H3                 |     1024 |         177 |  0.685904 |       1000.468927 |     1065.406780 |
| QBGF_SENTENCE_ONLY |     1024 |         177 |  0.684693 |       1017.728814 |     1082.666667 |
| FIXED_SENTENCE     |     1024 |         177 |  0.670398 |       1021.045198 |     1085.983051 |
| QBGF_FULL          |     1024 |         177 |  0.660413 |       1018.384181 |     1083.322034 |
| QBGF_PASSAGE_ONLY  |     1024 |         177 |  0.603107 |        966.451977 |     1031.389831 |
| QBGF_NO_QUERY      |     1024 |         177 |  0.542628 |       1020.231638 |     1085.169492 |
| NONE               |     1024 |         177 |  0.420662 |          0.000000 |       64.937853 |

Budget is selected-evidence tokens; final prompt cost includes framing/query. RAW_SKIP denotes SFR budget-matched whole-passage RAG; BM25_RAG_RAW_SKIP is the matched lexical retrieval control. The duplicate SFR extension alias is omitted here, not treated as independent evidence.


## 3. Point-estimate leaders are not the final policy

| dataset   |   budget | highest_point_estimate   |   best_quality |   FIXED_SENTENCE |
|:----------|---------:|:-------------------------|---------------:|-----------------:|
| HotpotQA  |      256 | H3                       |       0.425311 |         0.408247 |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY        |       0.452507 |         0.423310 |
| QASPER    |      256 | FIXED_SENTENCE           |       0.263430 |         0.263430 |
| QASPER    |     1024 | H3                       |       0.299487 |         0.298788 |
| SciFact   |      256 | C2                       |       0.710248 |         0.700062 |
| SciFact   |     1024 | BM25_RAG_RAW_SKIP        |       0.710716 |         0.670398 |

This six-cell map is useful for targeted diagnostic comparators and is already frozen for attribution. It is **not** a preregistered per-dataset adaptive final policy. Do not silently replace the existing global fallback selection rule with six post-hoc winners. Small differences here do not establish superiority without paired uncertainty.


## 4. Does QBGF qualify for promotion?

| dataset   |   budget |     point |    ci_low |   ci_high |   NI_margin | NI_established   |   QBGF_minus_fixed_prompt_tokens |   joint_holm_adjusted_p |
|:----------|---------:|----------:|----------:|----------:|------------:|:-----------------|---------------------------------:|------------------------:|
| QASPER    |      256 | -0.027915 | -0.040996 | -0.014908 |    0.003000 | False            |                        -2.540299 |                0.005099 |
| QASPER    |     1024 | -0.003734 | -0.012490 |  0.005008 |    0.003000 | False            |                        -2.537313 |                1.000000 |
| HotpotQA  |      256 |  0.003821 | -0.003511 |  0.011013 |    0.003000 | False            |                        -3.848358 |                1.000000 |
| HotpotQA  |     1024 |  0.009033 |  0.004544 |  0.013509 |    0.003000 | True             |                        -8.170929 |                0.022198 |
| SciFact   |      256 | -0.025094 | -0.087757 |  0.036542 |    0.010000 | False            |                        -2.706215 |                1.000000 |
| SciFact   |     1024 | -0.009985 | -0.071143 |  0.050456 |    0.010000 | False            |                        -2.661017 |                1.000000 |

Contrast is QBGF_FULL minus FIXED_SENTENCE. Promotion requires the CI lower bound to clear −0.003 F1 for QA and −0.01 macro-F1 for SciFact in **every** dataset/budget cell, with no greater mean prompt cost and at least one strict token gain. Only HotpotQA1024 establishes noninferiority; **five cells fail to establish it**. This is not a claim that QBGF is demonstrably harmful in all five: several intervals cross zero. QASPER256 has clear negative evidence (−0.027915 F1, CI[−0.040996,−0.014908]).

QBGF saves only about2.5–8.2 prompt tokens versus fixed sentences in these cells. Those small savings do not rescue the failed quality gate. HotpotQA1024 does improve over fixed sentences, but still loses to a strong passage baseline. Therefore QBGF should not be promoted as the universal final system.


## 5. Positive finding: query conditioning matters

| dataset   |   budget |    point |   ci_low |   ci_high |   joint_holm_adjusted_p |
|:----------|---------:|---------:|---------:|----------:|------------------------:|
| QASPER    |      256 | 0.088213 | 0.071636 |  0.105356 |                0.005099 |
| QASPER    |     1024 | 0.066681 | 0.051495 |  0.081844 |                0.005099 |
| HotpotQA  |      256 | 0.103475 | 0.095397 |  0.111746 |                0.005099 |
| HotpotQA  |     1024 | 0.020092 | 0.015356 |  0.024760 |                0.005099 |
| SciFact   |      256 | 0.187002 | 0.103136 |  0.270973 |                0.007799 |
| SciFact   |     1024 | 0.117784 | 0.055711 |  0.182024 |                0.011399 |

FULL beats NO_QUERY in all six cells with positive paired intervals and joint-Holm support. This tests query-conditioned allocation **within a query-dependent SFR retrieved pool**. It does not show that query-free retrieval was tested. The supported contribution is allocation/fidelity conditioning, not a universal fusion win.


## 6. Fusion itself remains unproven

| dataset   |   budget | right              |     point |    ci_low |   ci_high |   joint_holm_adjusted_p |
|:----------|---------:|:-------------------|----------:|----------:|----------:|------------------------:|
| QASPER    |      256 | QBGF_SENTENCE_ONLY | -0.000069 | -0.000581 |  0.000367 |                1.000000 |
| QASPER    |      256 | QBGF_PASSAGE_ONLY  |  0.066233 |  0.048862 |  0.083949 |                0.005099 |
| QASPER    |     1024 | QBGF_SENTENCE_ONLY | -0.002685 | -0.010267 |  0.004915 |                1.000000 |
| QASPER    |     1024 | QBGF_PASSAGE_ONLY  |  0.005474 | -0.005031 |  0.015607 |                1.000000 |
| HotpotQA  |      256 | QBGF_SENTENCE_ONLY | -0.000121 | -0.000363 |  0.000000 |                1.000000 |
| HotpotQA  |      256 | QBGF_PASSAGE_ONLY  |  0.018559 |  0.011475 |  0.025506 |                0.005099 |
| HotpotQA  |     1024 | QBGF_SENTENCE_ONLY |  0.000049 | -0.001157 |  0.001270 |                1.000000 |
| HotpotQA  |     1024 | QBGF_PASSAGE_ONLY  | -0.020163 | -0.025042 | -0.015425 |                0.005099 |
| SciFact   |      256 | QBGF_SENTENCE_ONLY |  0.000000 |  0.000000 |  0.000000 |                1.000000 |
| SciFact   |      256 | QBGF_PASSAGE_ONLY  |  0.217194 |  0.139681 |  0.298944 |                0.005099 |
| SciFact   |     1024 | QBGF_SENTENCE_ONLY | -0.024280 | -0.051337 | -0.004812 |                1.000000 |
| SciFact   |     1024 | QBGF_PASSAGE_ONLY  |  0.057305 | -0.009679 |  0.127139 |                1.000000 |

FULL-minus-SENTENCE_ONLY is near zero at256 and nonpositive on QASPER/SciFact at1024. These results do not establish mixed-granularity superiority, nor do nonsignificant intervals prove exact equivalence. FULL can beat passage-only under tight budgets while offering no clear advantage over sentence-only allocation.


## 7. External RAG and specialist context

| dataset   |   budget | right             |     point |    ci_low |   ci_high |   extension_holm_p |
|:----------|---------:|:------------------|----------:|----------:|----------:|-------------------:|
| QASPER    |      256 | BM25_RAG_RAW_SKIP |  0.082127 |  0.065689 |  0.099025 |           0.001200 |
| QASPER    |      256 | SFR_RAG_RAW_SKIP  |  0.066233 |  0.048862 |  0.083949 |           0.001200 |
| QASPER    |     1024 | BM25_RAG_RAW_SKIP |  0.039127 |  0.026233 |  0.052226 |           0.001200 |
| QASPER    |     1024 | SFR_RAG_RAW_SKIP  |  0.007098 | -0.003304 |  0.017170 |           0.886411 |
| HotpotQA  |      256 | BM25_RAG_RAW_SKIP |  0.032515 |  0.024094 |  0.041104 |           0.001200 |
| HotpotQA  |      256 | SFR_RAG_RAW_SKIP  |  0.005094 | -0.002327 |  0.012384 |           0.886411 |
| HotpotQA  |     1024 | BM25_RAG_RAW_SKIP | -0.003001 | -0.008673 |  0.002583 |           0.886411 |
| HotpotQA  |     1024 | SFR_RAG_RAW_SKIP  | -0.019789 | -0.024994 | -0.014683 |           0.001200 |
| SciFact   |      256 | BM25_RAG_RAW_SKIP |  0.237458 |  0.149720 |  0.328166 |           0.001200 |
| SciFact   |      256 | SFR_RAG_RAW_SKIP  |  0.218591 |  0.141282 |  0.300298 |           0.001200 |
| SciFact   |     1024 | BM25_RAG_RAW_SKIP | -0.050304 | -0.132752 |  0.031893 |           0.886411 |
| SciFact   |     1024 | SFR_RAG_RAW_SKIP  | -0.038566 | -0.121233 |  0.043969 |           0.886411 |

These supplemental comparisons use their own12-test Holm family; SFR extension contrasts repeat RAW_SKIP evidence. In particular, QBGF is significantly worse than SFR RAW_SKIP on HotpotQA1024 (−0.019789 F1). At256 it improves over passage controls on QASPER/SciFact, where whole-passage fitting and utilization are material confounds.

| method                     |   n_queries |       F1 |       EM |   prompt_tokens |
|:---------------------------|------------:|---------:|---------:|----------------:|
| SARA_CANONICAL_BM25_K5_N10 |        1005 | 0.395049 | 0.209950 |     1296.583085 |
| SARA_SFR_K5_N10            |        1005 | 0.436600 | 0.223881 |     1285.497512 |

SARA references are QASPER-only adapted Mistral systems with a different checkpoint, prompt and latent interface. Their higher F1 is a whole-system comparison, not proof that latent vectors are intrinsically superior. Historical k6 is not substituted into fresh VAL or confirmation tables.


## 8. Final freeze recommendation

Primary policy: **SFR top10 retrieval → FIXED_SENTENCE allocation → pinned Qwen3-8B BASE_FP16**, retaining both preregistered evidence budgets256/1024. No LoRA. The method is the **registered simple fallback**, not the winner of an invented global cross-dataset average. The final method remains a single policy, rather than choosing the best method separately by dataset.

QASPER has no fresh confirmation cohort. Future confirmation uses only the already reserved HotpotQA200 and SciFact50 IDs; historical QASPER TEST stays closed. Attribution remains a separate post-hoc150-VAL-query diagnostic and must not redefine primary claims or tune confirmation.


## 9. Part I readiness correction

The old freeze helper automatically enrolled every MAIN method and a qualified-only LLMLingua arm. This would have reintroduced skipped/unvalidated systems into confirmation. A separate reviewed helper (`src/week12_final_review.py`) now preserves original C/D and attribution code hashes and the selection rule, but retains only existing MAIN controls with complete BASE_FP16 VAL coverage at **both** budgets.

The explicit applicable confirmation list is: **NONE, RAW_SKIP, FIXED_SENTENCE, H3, QBGF_FULL**, all BASE_FP16 at256/1024. C2 remains a valid256 VAL control but is excluded from this uniform two-budget runner because C2-1024 was not evaluated. RAW_STRICT and unrestricted RAG secondary arms are unvalidated; LLMLingua/LoRA are not auto-enrolled. D-only ablations and C-EXT specialist systems remain diagnostic/reference conditions. No new method definition, frozen split or primary multiplicity family is changed.

Part I now reads H's saved tables instead of rerunning all bootstrap calculations. Its actual freeze still requires the entire frozen VAL membership, complete C-EXT, an explicit reason and matching original identities. Re-running an identical final freeze reuses the original artifact rather than changing its timestamp. No final artifact is written by report generation or a dry-run preview.


## 10. Exact Part I settings

Use the updated `week12_I_week13_freeze.ipynb`, set the flags below, then Run All. I reads saved VAL results and writes the final freeze; it does **not** start confirmation generation. `SMOKE_LIMIT` may remain3 and is unused here.

```python
CORE_FINAL = True
RUN_SMOKE = False
RUN_FULL = False
RESUME = True
OVERWRITE = False
RUN_REAL_MODEL_SMOKE = False
RUN_CANONICAL_SMOKE = False
ALLOW_LLMLINGUA_DOWNLOAD = False
RUN_LLMLINGUA_SMOKE = False
RUN_LLMLINGUA_FULL = False
RUN_TINY_LORA_SMOKE = False
RUN_FULL_LORA_TRAIN = False
FINAL_METHOD = "FIXED_SENTENCE"
FINAL_REASON = (
    "Retain the preregistered fixed-sentence fallback: QBGF does not establish "
    "noninferiority across all dataset/budget cells; mixed-granularity benefits "
    "are unproven. E/F/G were not run; retain no LoRA."
)
RETAIN_LORA = False
```

Expected primary method: FIXED_SENTENCE; reader BASE_FP16; retain_lora false. Review the explicit confirmation method list in section9 before manually executing I. The agent has not written the final freeze or started confirmation.


## 11. Interpretation limits and provenance

Paired95% intervals use10,000 paper/query/document-group resamples, seed42. p-values are paired cluster label-swap tests; Holm corrections are retained. Development exposure, many comparisons, SciFact sample size and QASPER mapping gaps limit claims. No unrun system is represented as an observed zero. Source support is mechanical provenance, not semantic faithfulness.

Source files: saved `qbgf/v3/analysis/{summary,primary_comparisons,fusion_diagnostic_comparisons}.csv`, C-EXT analysis tables, optional-phase decisions and immutable C/D result stores. Original scientific code hash: `0002987a3e2fe48f3d6511fc9fb409bba23a9a3594c3370e2de0e672c22afa95`. This report and the reviewed freeze helper do not alter historical outcomes or attribution query IDs.


## Verification

139 unit tests passed. Updated Part I safe-default notebook smoke passed. A real cached-data dry-run validated the fallback, all five eligible confirmation controls and JSON serialization without writing either final-freeze target. Original C/D scientific identities and frozen attribution IDs remain unchanged. No confirmation generation or final freeze was executed by the agent.

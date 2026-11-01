# Final Week 9 report

## Executive conclusion

MIXED_OR_NEGATIVE. Applied the frozen conjunction across both strong comparators and the separate conservative D1 compression criterion.

## Frozen protocol

TEST confirmation under immutable pretest manifest. Paper-cluster bootstrap: 10,000 resamples, seed 42. No tuning or retrospective promotion.

## Baselines

Table 1 is the compact primary comparison; Table 2 contains separately labelled extended ablations. Earlier FIT/DEV artifacts are never confirmation rows.

## W9 primary system

Canonical SARA first pass; FIXED-5/SEMANTIC@20; DraftSemantic coarse K=2; exact query-only MiniLM top-3 sentences in source order; F0 fresh regeneration inside unchanged SARA context.

## Final quality results

| system           | split   |   n_queries |       f1 |   exact_match | table    |
|:-----------------|:--------|------------:|---------:|--------------:|:---------|
| BM25_k8          | test    |        1309 | 0.423667 |      0.233766 | primary  |
| D0               | test    |        1309 | 0.376655 |      0.212376 | primary  |
| D1               | test    |        1309 | 0.373928 |      0.175707 | extended |
| W9-PRIMARY       | test    |        1309 | 0.372788 |      0.207028 | primary  |
| W9-SAFE          | test    |        1309 | 0.384465 |      0.220779 | primary  |
| W9-VERIFY        | test    |        1309 | 0.379775 |      0.212376 | primary  |
| original_SARA_G0 | test    |        1309 | 0.396357 |      0.222307 | primary  |
| semantic_k2      | test    |        1309 | 0.327922 |      0.183346 | extended |
| semantic_k4      | test    |        1309 | 0.376655 |      0.212376 | primary  |
| semantic_k5      | test    |        1309 | 0.393448 |      0.220779 | extended |
| semantic_k6      | test    |        1309 | 0.406446 |      0.229947 | primary  |
| semantic_k8      | test    |        1309 | 0.415539 |      0.227655 | primary  |

## Statistical comparisons

| left       | right            | family      | status     |    delta_f1 |      ci_low |     ci_high | cluster   |   resamples |   seed |
|:-----------|:-----------------|:------------|:-----------|------------:|------------:|------------:|:----------|------------:|-------:|
| W9-PRIMARY | D0               | primary     | COMPATIBLE | -0.00386711 | -0.0200741  |  0.0122498  | paper_id  |       10000 |     42 |
| W9-PRIMARY | semantic_k4      | primary     | COMPATIBLE | -0.00386711 | -0.0200741  |  0.0122498  | paper_id  |       10000 |     42 |
| W9-PRIMARY | semantic_k6      | primary     | COMPATIBLE | -0.0336587  | -0.0510873  | -0.016226   | paper_id  |       10000 |     42 |
| W9-PRIMARY | semantic_k8      | primary     | COMPATIBLE | -0.0427516  | -0.0603753  | -0.024722   | paper_id  |       10000 |     42 |
| W9-PRIMARY | original_SARA_G0 | primary     | COMPATIBLE | -0.0235693  | -0.0402687  | -0.00683141 | paper_id  |       10000 |     42 |
| W9-PRIMARY | D1               | mechanistic | COMPATIBLE | -0.00113998 | -0.0174961  |  0.0156178  | paper_id  |       10000 |     42 |
| W9-SAFE    | W9-PRIMARY       | secondary   | COMPATIBLE |  0.0116771  | -0.00401753 |  0.0271305  | paper_id  |       10000 |     42 |
| W9-VERIFY  | W9-PRIMARY       | secondary   | COMPATIBLE |  0.00698703 | -0.0033623  |  0.0176461  | paper_id  |       10000 |     42 |
| W9-VERIFY  | D0               | secondary   | COMPATIBLE |  0.00311992 | -0.00920798 |  0.0155834  | paper_id  |       10000 |     42 |
| W9-VERIFY  | W9-SAFE          | secondary   | COMPATIBLE | -0.00469002 | -0.0168516  |  0.00762329 | paper_id  |       10000 |     42 |

## Cost results

| system           |   final_prompt_tokens |   total_query_prompt_tokens |   total_query_generated_tokens |   num_lm_calls |   evidence_tokens |   retrieval_query_time |   selection_latency |   extraction_latency |   revision_latency |   verifier_latency |   sentence_index_conceptual_seconds |   query_embedding_conceptual_seconds |   embedding_lookup_compute_latency |   sentence_ranking_latency | measured_end_to_end_latency   | cost_note                                                                                                      |
|:-----------------|----------------------:|----------------------------:|-------------------------------:|---------------:|------------------:|-----------------------:|--------------------:|---------------------:|-------------------:|-------------------:|------------------------------------:|-------------------------------------:|-----------------------------------:|---------------------------:|:------------------------------|:---------------------------------------------------------------------------------------------------------------|
| BM25_k8          |              1779.66  |                    1779.66  |                        13.2674 |        1       |           nan     |              nan       |          nan        |                  nan |           4.62548  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| D0               |               929.807 |                     929.807 |                        14.8793 |        1       |           nan     |              nan       |          nan        |                  nan |           2.56928  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| D1               |              3026.83  |                    3956.64  |                        51.7525 |        2       |          2050.91  |                0.42689 |            0.427961 |                  nan |           2.07376  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| W9-PRIMARY       |              1071.24  |                    2001.05  |                        36.3254 |        2       |            95.317 |                0.42689 |            0.427961 |                    0 |           0.958749 |       nan          |                            0.325756 |                            0.0083792 |                           0.196002 |                 0.00138368 |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| W9-SAFE          |              1163.06  |                    2092.87  |                        29.3018 |        2       |            95.317 |                0.42689 |            0.427961 |                    0 |           0.750969 |       nan          |                            0.325756 |                            0.0083792 |                           0.196002 |                 0.00138368 |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| W9-VERIFY        |              1071.24  |                    2001.05  |                        36.3254 |        2       |            95.317 |                0.42689 |            0.427961 |                    0 |           0.958749 |         0.00276359 |                            0.325756 |                            0.0083792 |                           0.196002 |                 0.00138368 |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| original_SARA_G0 |              1012.61  |                    1403.53  |                        20.0176 |        1.42017 |           nan     |              nan       |            0.513405 |                  nan |           1.22478  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| semantic_k2      |               518.257 |                     518.257 |                        18.2193 |        1       |           nan     |              nan       |          nan        |                  nan |           1.77947  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| semantic_k4      |               929.807 |                     929.807 |                        14.8793 |        1       |           nan     |              nan       |          nan        |                  nan |           2.56928  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| semantic_k5      |              1140.77  |                    1140.77  |                        15.0787 |        1       |           nan     |              nan       |          nan        |                  nan |           3.07639  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| semantic_k6      |              1348.18  |                    1348.18  |                        15.0963 |        1       |           nan     |              nan       |          nan        |                  nan |           3.5785   |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |
| semantic_k8      |              1775.65  |                    1775.65  |                        14.7395 |        1       |           nan     |              nan       |          nan        |                  nan |           4.63467  |       nan          |                          nan        |                          nan         |                         nan        |               nan          |                               | Null means unmeasured/unestablished. Cached work retains conceptual attribution; no summed end-to-end latency. |

## Safety / HELP / HURT results

| system           |      help |      hurt |   answer_changed |   preservation |   mean_positive_delta |   mean_negative_delta |   f0_help_retention |   f0_hurt_rescue | retention_definition                                |   f0_hurt_rejection |
|:-----------------|----------:|----------:|-----------------:|---------------:|----------------------:|----------------------:|--------------------:|-----------------:|:----------------------------------------------------|--------------------:|
| BM25_k8          | 0.219251  | 0.152788  |        0.550038  |       0.449962 |              0.415289 |             -0.288473 |         nan         |       nan        | nan                                                 |          nan        |
| D0               | 0         | 0         |        0         |       1        |            nan        |            nan        |         nan         |       nan        | nan                                                 |          nan        |
| D1               | 0.315508  | 0.278839  |        0.764706  |       0.235294 |              0.299812 |             -0.349021 |         nan         |       nan        | nan                                                 |          nan        |
| W9-PRIMARY       | 0.185638  | 0.189458  |        0.590527  |       0.409473 |              0.333718 |             -0.347215 |         nan         |       nan        | nan                                                 |          nan        |
| W9-SAFE          | 0.0229183 | 0.0145149 |        0.0618793 |       0.938121 |              0.537592 |             -0.310357 |           0.0864198 |         0.951613 | Still HELP vs D0; rescue means no longer HURT vs D0 |          nan        |
| W9-VERIFY        | 0.078686  | 0.0687548 |        0.202445  |       0.797555 |              0.429368 |             -0.446047 |           0.423868  |         0.637097 | Still HELP vs D0; rescue means no longer HURT vs D0 |            0.637097 |
| original_SARA_G0 | 0.0603514 | 0.0275019 |        0.161192  |       0.838808 |              0.452397 |             -0.276264 |         nan         |       nan        | nan                                                 |          nan        |
| semantic_k2      | 0.121467  | 0.222307  |        0.553858  |       0.446142 |              0.303886 |             -0.385225 |         nan         |       nan        | nan                                                 |          nan        |
| semantic_k4      | 0         | 0         |        0         |       1        |            nan        |            nan        |         nan         |       nan        | nan                                                 |          nan        |
| semantic_k5      | 0.118411  | 0.0855615 |        0.34301   |       0.65699  |              0.322819 |             -0.250592 |         nan         |       nan        | nan                                                 |          nan        |
| semantic_k6      | 0.161192  | 0.113827  |        0.427044  |       0.572956 |              0.367361 |             -0.258477 |         nan         |       nan        | nan                                                 |          nan        |
| semantic_k8      | 0.203972  | 0.135982  |        0.521772  |       0.478228 |              0.385358 |             -0.292214 |         nan         |       nan        | nan                                                 |          nan        |

## Evidence compression analysis

{
  "raw_selected_chunk_tokens": 2050.910618792972,
  "e3_evidence_tokens": 95.3170359052712,
  "retained_fraction": 0.04647547047241308,
  "evidence_suppression": 0.9535245295275869,
  "final_prompt_reduction": 0.6460854601797117,
  "cumulative_prompt_reduction": 0.4942560138614852,
  "raw_lm_calls": 2.0,
  "compact_lm_calls": 2.0,
  "f1_delta_vs_d1": -0.0011399822976350116,
  "help_rate_delta_vs_d1": -0.12987012987012986,
  "hurt_rate_delta_vs_d1": -0.08938120702826585,
  "interpretation": "Evidence exposure, final prompt, cumulative prompt and LM calls are different quantities."
}

## Controller result

G2 is secondary and post-generation. KEEP pays the full F0 cost. Its deployment threshold remains 0.70.

## Mechanistic findings

FIT/DEV development findings: sentence suppression produced compact exact-source evidence; sentence-count surfaces and oracle ceilings are retrospective, not deployable policies. Confirmation comparisons above determine transfer.

## Negative findings

Earlier generative extraction was brittle and did not establish a deployable gain. Nested FIT verifier estimates did not convincingly beat F0/F2-A; pooled deployment FIT scores are not honest OOF estimates.

## Error analysis

Deterministic category samples saved only after aggregate seal; explanatory only. No method changes permitted.

## Final hit/fail classification

MIXED_OR_NEGATIVE

## Limitations

- FIT/DEV findings are development evidence
- Evidence exposure reduction is not an equal reduction in end-to-end compute
- One frozen canonical generator; no new robustness panel
- Raw bootstrap CIs; no multiplicity-adjusted significance claim

## What W9 establishes

Only the measured frozen-system effects, costs and intervals on this confirmation split; development findings remain separately labelled.

## What W9 does NOT establish

No general RAG superiority, causal evidence identification, dynamic-k success, or proportional end-to-end speedup from evidence suppression.

## Recommended next step after Week 9

Close Week 9 with the frozen result. Carry limitations into a separately preregistered Week 10 study; do not retune on this TEST split.

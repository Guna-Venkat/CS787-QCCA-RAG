# Week 8 — Part-Wise Analysis

## Week objective

Investigate draft-aware selective evidence promotion, measured generator interaction and adaptation while separating real execution from archived/smoke prototypes.

# Part A — Promotion response surface

**Status: COMPLETED**

## 1. Research question

Is there selective evidence-promotion headroom beyond a semantic-k4 draft?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/surfaces/promotion_surface_fit.jsonl](../../week8/surfaces/promotion_surface_fit.jsonl)

## 5. Main numerical results

### Raw six-candidate FIT surface recomputation

|     queries |   candidate_rows |   base_F1 |   always_best_candidate_F1 |   retrospective_selective_oracle_F1 |
|------------:|-----------------:|----------:|---------------------------:|------------------------------------:|
| 1746.000000 |     10476.000000 |  0.385631 |                   0.481899 |                            0.485209 |

Source: [results/week8/surfaces/promotion_surface_fit.jsonl](../../week8/surfaces/promotion_surface_fit.jsonl).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

FIT has10476 candidate rows for1746 queries. Base F1 is 0.385631; the retrospective selective oracle is 0.485209.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Promotion oracle uses known outcome scores; it does not establish deployable selective gains.

## 9. Part conclusion

- FIT has10476 candidate rows for1746 queries. Base F1 is 0.385631; the retrospective selective oracle is 0.485209.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part B — Draft uncertainty and self-consistency

**Status: PARTIALLY EXECUTED**

## 1. Research question

Can draft generation signals identify when more evidence may help?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Notebook code cells2–5 have generation, feature and self-consistency outputs; the actual JSONL is primary and is kept separate from the old notebook display.

## 4. Result artifacts used

- [notebooks/week8/week8b_draft_diagnostics.ipynb](../../../notebooks/week8/week8b_draft_diagnostics.ipynb)
- [results/week8/drafts/draft_diagnostics_internal_val.jsonl](../../week8/drafts/draft_diagnostics_internal_val.jsonl)

## 5. Main numerical results

No completed result artifact was found.

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

Executed code-cell outputs record342 generated drafts and12 uncertainty features. The saved output also has degenerate logprob/length features (e.g. answer length mean0) despite completed drafts.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

SOURCE CONFLICT: historical notebook features include constants/placeholder-like values; later measured-token gate inputs must retain their distinct provenance. Do not endorse the old classifier metrics as clean uncertainty evidence.

## 9. Part conclusion

- Executed code-cell outputs record342 generated drafts and12 uncertainty features. The saved output also has degenerate logprob/length features (e.g. answer length mean0) despite completed drafts.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part C — Candidate utility modeling

**Status: COMPLETED**

## 1. Research question

Can a learned selector outperform draft-semantic or heuristic selection?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/w8c_utility/fit_oof_comparison.csv](../../week8/w8c_utility/fit_oof_comparison.csv)
- [results/week8/w8c_utility/w8c_frozen_selector_config.json](../../week8/w8c_utility/w8c_frozen_selector_config.json)

## 5. Main numerical results

### fit oof comparison

| selector         |   base_k4_f1 |   oracle_selective1_f1 |   num_queries |   top1_oracle_accuracy |   near_oracle_selection_rate |   mean_selected_delta_f1 |   unconditional_downstream_f1 |   num_promoted |   mean_regret |   median_regret |   help_precision |   hurt_rate |   final_downstream_f1 |   selector_headroom_recovery |
|:-----------------|-------------:|-----------------------:|--------------:|-----------------------:|-----------------------------:|-------------------------:|------------------------------:|---------------:|--------------:|----------------:|-----------------:|------------:|----------------------:|-----------------------------:|
| S0 Rank5         |     0.385631 |               0.485174 |   1746.000000 |               0.735395 |                     0.742268 |                 0.012389 |                      0.398021 |    1746.000000 |      0.083878 |        0.000000 |         0.129439 |    0.094502 |              0.417425 |                     0.319403 |
| S1 BM25          |     0.385631 |               0.485174 |   1746.000000 |               0.727377 |                     0.734250 |                 0.017482 |                      0.403113 |    1746.000000 |      0.078785 |        0.000000 |         0.130011 |    0.101375 |              0.421979 |                     0.365144 |
| S2 Semantic      |     0.385631 |               0.485174 |   1746.000000 |               0.735395 |                     0.742268 |                 0.012389 |                      0.398021 |    1746.000000 |      0.083878 |        0.000000 |         0.129439 |    0.094502 |              0.417425 |                     0.319403 |
| S3 DraftSemantic |     0.385631 |               0.485174 |   1746.000000 |               0.741695 |                     0.749141 |                 0.019823 |                      0.405454 |    1746.000000 |      0.076444 |        0.000000 |         0.143184 |    0.098511 |              0.424430 |                     0.389770 |
| S4 MIG           |     0.385631 |               0.485174 |   1746.000000 |               0.744559 |                     0.750859 |                 0.017753 |                      0.403384 |    1746.000000 |      0.078514 |        0.000000 |         0.138603 |    0.092784 |              0.422394 |                     0.369316 |
| Oracle           |     0.385631 |               0.485174 |   1746.000000 |               1.000000 |                     1.000000 |                 0.096267 |                      0.481899 |    1746.000000 |      0.000000 |        0.000000 |         0.302405 |    0.015464 |              0.485174 |                     1.000000 |
| U0               |     0.385631 |               0.485174 |   1746.000000 |               0.724513 |                     0.730813 |                 0.010577 |                      0.396208 |    1746.000000 |      0.085691 |        0.000000 |         0.115120 |    0.101375 |              0.413699 |                     0.281970 |
| U1               |     0.385631 |               0.485174 |   1746.000000 |               0.721649 |                     0.729095 |                 0.012770 |                      0.398401 |    1746.000000 |      0.083498 |        0.000000 |         0.119129 |    0.102520 |              0.419636 |                     0.341613 |
| U2-F             |     0.385631 |               0.485174 |   1746.000000 |               0.718213 |                     0.725659 |                 0.012201 |                      0.397832 |    1746.000000 |      0.084067 |        0.000000 |         0.122566 |    0.108247 |              0.419232 |                     0.337552 |
| U2-A             |     0.385631 |               0.485174 |   1746.000000 |               0.737113 |                     0.745132 |                 0.016968 |                      0.402600 |    1746.000000 |      0.079299 |        0.000000 |         0.139748 |    0.100229 |              0.422009 |                     0.365444 |
| U3               |     0.385631 |               0.485174 |   1746.000000 |               0.725086 |                     0.732532 |                 0.012090 |                      0.397721 |    1746.000000 |      0.084177 |        0.000000 |         0.115120 |    0.092784 |              0.413827 |                     0.283256 |
| U4-A             |     0.385631 |               0.485174 |   1746.000000 |               0.741695 |                     0.748568 |                 0.017139 |                      0.402770 |    1746.000000 |      0.079128 |        0.000000 |         0.145475 |    0.103093 |              0.424463 |                     0.390104 |

Source: [results/week8/w8c_utility/fit_oof_comparison.csv](../../week8/w8c_utility/fit_oof_comparison.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

U4-A FIT OOF oracle-gated headroom recovery is0.390104 versus DraftSemantic0.389770: an extremely small difference. Oracle-gated F1 must not be mistaken for unconditional promoted-answer quality.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Many neutral/tied candidates inflate top1 oracle accuracy. Retrospective gate and candidate utility selection answer different questions.

## 9. Part conclusion

- U4-A FIT OOF oracle-gated headroom recovery is0.390104 versus DraftSemantic0.389770: an extremely small difference. Oracle-gated F1 must not be mistaken for unconditional promoted-answer quality.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part D — Measured generator interaction features

**Status: COMPLETED**

## 1. Research question

Do semantic/redundancy/masking features improve selection after draft generation?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/w8d_attribution/full/features_fit.csv](../../week8/w8d_attribution/full/features_fit.csv)
- [results/week8/w8d_closeout/internal_val_metrics.csv](../../week8/w8d_closeout/internal_val_metrics.csv)
- [results/week8/w8d_closeout/closeout_summary.json](../../week8/w8d_closeout/closeout_summary.json)

## 5. Main numerical results

### Masking feature availability

|   rows |   measured_masking |   missing_masking |
|-------:|-------------------:|------------------:|
|  10476 |              10464 |                12 |

Source: [results/week8/w8d_attribution/full/features_fit.csv](../../week8/w8d_attribution/full/features_fit.csv).

### internal val metrics

| selector   |   always_promote_f1 |   base_f1 |   oracle_gated_shr |   n |   help_rate |   hurt_rate |   mean_delta |   mean_regret |   top1_tied_best |   near_oracle_rate |   final_prompt_tokens |   cumulative_prompt_tokens |   escalation.n |   escalation.help_rate |   escalation.hurt_rate |   escalation.mean_delta |   escalation.mean_regret |   escalation.top1_tied_best |   escalation.near_oracle_rate |
|:-----------|--------------------:|----------:|-------------------:|----:|------------:|------------:|-------------:|--------------:|-----------------:|-------------------:|----------------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|-------------------------:|----------------------------:|------------------------------:|
| S3         |            0.412572 |  0.389061 |           0.447599 | 342 |    0.146199 |    0.099415 |     0.023511 |      0.067451 |         0.769006 |           0.771930 |           1125.228070 |                2052.385965 |             98 |               0.510204 |               0.153061 |                0.143096 |                 0.176601 |                    0.377551 |                      0.377551 |
| S5         |            0.389016 |  0.389061 |           0.205557 | 342 |    0.111111 |    0.111111 |    -0.000045 |      0.091007 |         0.730994 |           0.739766 |           1134.005848 |                2061.163743 |             98 |               0.387755 |               0.173469 |                0.065716 |                 0.253981 |                    0.285714 |                      0.285714 |
| S6         |            0.396327 |  0.389061 |           0.216938 | 342 |    0.108187 |    0.099415 |     0.007266 |      0.083697 |         0.730994 |           0.739766 |           1116.874269 |                2044.032164 |             98 |               0.377551 |               0.163265 |                0.069354 |                 0.250343 |                    0.265306 |                      0.265306 |
| S7         |            0.412572 |  0.389061 |           0.447599 | 342 |    0.146199 |    0.099415 |     0.023511 |      0.067451 |         0.769006 |           0.771930 |           1125.605263 |                2052.763158 |             98 |               0.510204 |               0.153061 |                0.143096 |                 0.176601 |                    0.377551 |                      0.377551 |
| S8         |            0.412572 |  0.389061 |           0.447599 | 342 |    0.146199 |    0.099415 |     0.023511 |      0.067451 |         0.769006 |           0.771930 |           1125.605263 |                2052.763158 |             98 |               0.510204 |               0.153061 |                0.143096 |                 0.176601 |                    0.377551 |                      0.377551 |
| U4-A       |            0.408329 |  0.389061 |           0.286666 | 342 |    0.134503 |    0.070175 |     0.019267 |      0.071695 |         0.777778 |           0.786550 |           1164.154971 |                2091.312865 |             98 |               0.469388 |               0.132653 |                0.091646 |                 0.228051 |                    0.336735 |                      0.346939 |
| U4-B       |            0.416716 |  0.389061 |           0.395678 | 342 |    0.134503 |    0.064327 |     0.027655 |      0.063308 |         0.769006 |           0.783626 |           1139.616959 |                2066.774854 |             98 |               0.469388 |               0.091837 |                0.126497 |                 0.193200 |                    0.346939 |                      0.357143 |

Source: [results/week8/w8d_closeout/internal_val_metrics.csv](../../week8/w8d_closeout/internal_val_metrics.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

Masking measured10464/10476 FIT candidate rows; missing12 are not observed zeros. On INTERNAL-VAL, S3/S7 always-promote F1 is0.412572 versus base0.389061; U4-A is0.408329.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Exact source masking is an input intervention, not proof of semantic faithfulness or internal reasoning. Development/OOF selector selection is separate from held-out gate performance.

## 9. Part conclusion

- Masking measured10464/10476 FIT candidate rows; missing12 are not observed zeros. On INTERNAL-VAL, S3/S7 always-promote F1 is0.412572 versus base0.389061; U4-A is0.408329.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part D-CLOSEOUT — Selector replay and frozen decisions

**Status: COMPLETED**

## 1. Research question

Does a replay preserve selector identities and distinguish actual from oracle-gated improvement?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/w8d_closeout/fit_metrics.csv](../../week8/w8d_closeout/fit_metrics.csv)
- [results/week8/w8d_closeout/fit_frozen_decisions.json](../../week8/w8d_closeout/fit_frozen_decisions.json)

## 5. Main numerical results

### fit metrics

| selector    |   always_promote_f1 |   base_f1 |   oracle_gated_shr |    n |   help_rate |   hurt_rate |   mean_delta |   mean_regret |   top1_tied_best |   near_oracle_rate |   final_prompt_tokens |   cumulative_prompt_tokens |   escalation.n |   escalation.help_rate |   escalation.hurt_rate |   escalation.mean_delta |   escalation.mean_regret |   escalation.top1_tied_best |   escalation.near_oracle_rate |
|:------------|--------------------:|----------:|-------------------:|-----:|------------:|------------:|-------------:|--------------:|-----------------:|-------------------:|----------------------:|---------------------------:|---------------:|-----------------------:|-----------------------:|------------------------:|-------------------------:|----------------------------:|------------------------------:|
| S3          |            0.405454 |  0.385631 |           0.389770 | 1746 |    0.143184 |    0.098511 |     0.019823 |      0.076444 |         0.741695 |           0.749141 |           1138.769187 |                2071.313288 |            528 |               0.473485 |               0.106061 |                0.128301 |                 0.200869 |                    0.344697 |                      0.354167 |
| S5          |            0.396207 |  0.385631 |           0.292828 | 1746 |    0.113975 |    0.095647 |     0.010576 |      0.085691 |         0.723368 |           0.726804 |           1140.276632 |                2072.820733 |            528 |               0.376894 |               0.107955 |                0.096390 |                 0.232780 |                    0.272727 |                      0.276515 |
| S6          |            0.403384 |  0.385631 |           0.369316 | 1746 |    0.138603 |    0.092784 |     0.017753 |      0.078514 |         0.744559 |           0.750859 |           1126.317297 |                2058.861397 |            528 |               0.458333 |               0.107955 |                0.121568 |                 0.207602 |                    0.333333 |                      0.340909 |
| S7          |            0.405454 |  0.385631 |           0.389770 | 1746 |    0.143184 |    0.098511 |     0.019823 |      0.076444 |         0.741695 |           0.749141 |           1138.622566 |                2071.166667 |            528 |               0.473485 |               0.106061 |                0.128301 |                 0.200869 |                    0.344697 |                      0.354167 |
| S8          |            0.400908 |  0.385631 |           0.372510 | 1746 |    0.144903 |    0.105384 |     0.015277 |      0.080990 |         0.738259 |           0.746850 |           1136.412944 |                2068.957045 |            528 |               0.479167 |               0.117424 |                0.122619 |                 0.206551 |                    0.348485 |                      0.361742 |
| U4-A        |            0.402770 |  0.385631 |           0.390104 | 1746 |    0.145475 |    0.103093 |     0.017139 |      0.079128 |         0.741695 |           0.748568 |           1160.022337 |                2092.566438 |            528 |               0.481061 |               0.106061 |                0.128411 |                 0.200759 |                    0.354167 |                      0.365530 |
| U4-A-replay |            0.402770 |  0.385631 |           0.390104 | 1746 |    0.145475 |    0.103093 |     0.017139 |      0.079128 |         0.741695 |           0.748568 |           1160.022337 |                2092.566438 |            528 |               0.481061 |               0.106061 |                0.128411 |                 0.200759 |                    0.354167 |                      0.365530 |
| U4-B        |            0.403255 |  0.385631 |           0.395365 | 1746 |    0.142039 |    0.101375 |     0.017624 |      0.078643 |         0.735395 |           0.741695 |           1157.974227 |                2090.518328 |            528 |               0.469697 |               0.104167 |                0.130142 |                 0.199028 |                    0.339015 |                      0.348485 |

Source: [results/week8/w8d_closeout/fit_metrics.csv](../../week8/w8d_closeout/fit_metrics.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

Separate FIT metrics and frozen-decision artifacts exist for the closeout. Always-promote and oracle-gated SHR columns are reported separately.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Replay/derived closeout is not a new independent cohort; no additional confirmation is implied.

## 9. Part conclusion

- Separate FIT metrics and frozen-decision artifacts exist for the closeout. Always-promote and oracle-gated SHR columns are reported separately.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part E — Measured uncertainty controller

**Status: COMPLETED**

## 1. Research question

Can an implementable gate retain helpful promotions while rejecting harmful ones?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/w8e_controller/internal_val_metrics.csv](../../week8/w8e_controller/internal_val_metrics.csv)
- [results/week8/w8e_controller/controller_summary.json](../../week8/w8e_controller/controller_summary.json)
- [results/week8/w8e_controller/pipeline_status.json](../../week8/w8e_controller/pipeline_status.json)

## 5. Main numerical results

### internal val metrics

| Unnamed: 0       |   num_queries |   promotion_rate |   downstream_f1 |   gain_over_k4 |   mean_final_prompt_tokens |   mean_cumulative_prompt_tokens |   help_rate_all_queries |   hurt_rate_all_queries |
|:-----------------|--------------:|-----------------:|----------------:|---------------:|---------------------------:|--------------------------------:|------------------------:|------------------------:|
| E0               |           342 |         0.000000 |        0.389061 |       0.000000 |                 935.157895 |                      935.157895 |                0.000000 |                0.000000 |
| E1               |           342 |         1.000000 |        0.412572 |       0.023511 |                1133.228070 |                     2068.385965 |                0.146199 |                0.099415 |
| E2               |           342 |         0.625731 |        0.406122 |       0.017061 |                1060.959064 |                     1651.935673 |                0.131579 |                0.087719 |
| E3               |           342 |         0.625731 |        0.401963 |       0.012902 |                1084.216374 |                     1675.192982 |                0.122807 |                0.070175 |
| E5-DraftSemantic |           342 |         0.421053 |        0.400080 |       0.011018 |                1017.593567 |                     1416.111111 |                0.055556 |                0.046784 |
| E5-U4-A          |           342 |         0.485380 |        0.399381 |       0.010320 |                1050.204678 |                     1509.903509 |                0.087719 |                0.035088 |

Source: [results/week8/w8e_controller/internal_val_metrics.csv](../../week8/w8e_controller/internal_val_metrics.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

The 342-query INTERNAL-VAL table distinguishes no-promotion E0 F10.389061, always-promotion E1 0.412572, and gated E2 0.406122. Pipeline status is complete for this controller artifact set.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Latency is explicitly an estimate of recorded components, not measured end-to-end deployment latency.

## 9. Part conclusion

- The 342-query INTERNAL-VAL table distinguishes no-promotion E0 F10.389061, always-promotion E1 0.412572, and gated E2 0.406122. Pipeline status is complete for this controller artifact set.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part F — Query-conditioned latent adaptation

**Status: COMPLETED**

## 1. Research question

Does conditioning/adapting the generator improve draft answers?

## 2. Experimental design

| component          | value                                                                      |
|:-------------------|:---------------------------------------------------------------------------|
| dataset            | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks    | 5..10 beyond semantic k4 draft                                             |
| reader             | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder   | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed               | 42                                                                         |
| validation_queries | 342                                                                        |
| Role               | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

Completed machine-readable output is present for the named conditions; coverage is limited to the recorded rows and checks below.

## 4. Result artifacts used

- [results/week8/w8f_qcla/full/comparison.csv](../../week8/w8f_qcla/full/comparison.csv)
- [results/week8/w8f_qcla/full/summary.json](../../week8/w8f_qcla/full/summary.json)

## 5. Main numerical results

### comparison

| variant   |   validation_queries |   trainable_params |   validation_f1 |   validation_query_mean_answer_nll |   mean_final_prompt_tokens |   mean_cumulative_prompt_tokens |   mean_adapted_generation_latency_sec |   mean_estimated_cumulative_generation_latency_sec |   mean_conditioning_encode_seconds |   end_to_end_latency_sec |
|:----------|---------------------:|-------------------:|----------------:|-----------------------------------:|---------------------------:|--------------------------------:|--------------------------------------:|---------------------------------------------------:|-----------------------------------:|-------------------------:|
| qcla_0    |                  342 |                  0 |        0.389061 |                           1.259202 |                 935.157895 |                      935.157895 |                              2.711161 |                                           2.711161 |                           0.000000 |                      nan |
| qcla_1    |                  342 |           52439552 |        0.393799 |                           1.218129 |                 935.157895 |                      935.157895 |                              2.523085 |                                           2.523085 |                           0.102399 |                      nan |
| qcla_2    |                  342 |           62925312 |        0.398994 |                           1.219554 |                 935.157895 |                     1870.315789 |                              2.555450 |                                           5.247229 |                           0.204798 |                      nan |

Source: [results/week8/w8f_qcla/full/comparison.csv](../../week8/w8f_qcla/full/comparison.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

QCLA0/1/2 validation F1 is0.389061/0.393799/0.398994. QCLA2 doubles recorded cumulative prompt tokens; trainable parameters are about62.9M.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

No paired quality-superiority interval is established by the comparison CSV alone. Training/checkpoint changes make this an adapted-system comparison.

## 9. Part conclusion

- QCLA0/1/2 validation F1 is0.389061/0.393799/0.398994. QCLA2 doubles recorded cumulative prompt tokens; trainable parameters are about62.9M.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part G — Outcome-weighted SFT

**Status: IMPLEMENTED**

## 1. Research question

Does outcome-weighted generator adaptation improve difficult cases?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

The status is based on the available artifacts, not notebook existence alone. No completed result artifact was found.

## 4. Result artifacts used

- [notebooks/week8/week8g_outcome_weighted_sft.ipynb](../../../notebooks/week8/week8g_outcome_weighted_sft.ipynb)

## 5. Main numerical results

No completed result artifact was found.

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

No executed code outputs or completed full-run result artifact were found in the named G notebook/output paths.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Model/training scaffold presence is not completed training or evaluation.

## 9. Part conclusion

- No executed code outputs or completed full-run result artifact were found in the named G notebook/output paths.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part H — Full-system integration

**Status: SMOKE-TESTED**

## 1. Research question

Does the integrated DAFiR system improve answer quality and cumulative cost?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

The status is based on the available artifacts, not notebook existence alone. Available outputs must be interpreted under the status shown.

## 4. Result artifacts used

- [results/week8/tables/w8h_dafir1_cost_summary_smoke.csv](../../week8/tables/w8h_dafir1_cost_summary_smoke.csv)

## 5. Main numerical results

### w8h dafir1 cost summary smoke

|   mean_final_context_tokens |   mean_cumulative_prompt_tokens |   mean_generated_tokens |   mean_e2e_latency_sec |   mean_num_passes |   promotion_rate |   mean_num_promotions |   gpu_peak_memory_mb |
|----------------------------:|--------------------------------:|------------------------:|-----------------------:|------------------:|-----------------:|----------------------:|---------------------:|
|                 1125.500000 |                     1944.500000 |            -1946.000000 |               4.514522 |          2.000000 |         1.000000 |              1.000000 |             0.000000 |

Source: [results/week8/tables/w8h_dafir1_cost_summary_smoke.csv](../../week8/tables/w8h_dafir1_cost_summary_smoke.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

Only explicitly smoke-labelled full-system cost output was located; full-system completion is not established.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

Do not upgrade smoke output to a final scientific evaluation.

## 9. Part conclusion

- Only explicitly smoke-labelled full-system cost output was located; full-system completion is not established.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part I — Ablation scaffold

**Status: IMPLEMENTED**

## 1. Research question

Which components contribute in the integrated system?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

The status is based on the available artifacts, not notebook existence alone. No completed result artifact was found.

## 4. Result artifacts used

- [notebooks/week8/week8i_ablations.ipynb](../../../notebooks/week8/week8i_ablations.ipynb)

## 5. Main numerical results

No completed result artifact was found.

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

No completed result artifact was found for the named I full-run path.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

No component contribution is inferred from code alone.

## 9. Part conclusion

- No completed result artifact was found for the named I full-run path.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Part J — Final DEV evaluation scaffold

**Status: SMOKE-TESTED**

## 1. Research question

Does the frozen integrated system generalize to DEV?

## 2. Experimental design

| component        | value                                                                      |
|:-----------------|:---------------------------------------------------------------------------|
| dataset          | QASPER TRAIN-derived FIT1746 / INTERNAL-VAL342; DEV231 where named         |
| candidate_ranks  | 5..10 beyond semantic k4 draft                                             |
| reader           | Historical SARA/Mistral checkpoint                                         |
| semantic_encoder | Canonical MiniLM for routing; SFR for historical SARA embedding interface  |
| seed             | 42                                                                         |
| Role             | Protocol/context only; outcomes are accepted only from the artifacts below |

## 3. What was implemented / executed

The status is based on the available artifacts, not notebook existence alone. Available outputs must be interpreted under the status shown.

## 4. Result artifacts used

- [notebooks/week8/week8j_final_dev_evaluation.ipynb](../../../notebooks/week8/week8j_final_dev_evaluation.ipynb)
- [results/week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv](../../week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv)

## 5. Main numerical results

### w8j bootstrap dafir vs k4 smoke

| comparison                |   n_resamples |   n_queries |   n_papers |   mean_treatment_f1 |   mean_control_f1 |   delta_f1 | ci_f1_95                                     |   p_value_f1 |   delta_final_tokens | ci_final_tokens_95   |   delta_cum_tokens |   delta_latency_sec |
|:--------------------------|--------------:|------------:|-----------:|--------------------:|------------------:|-----------:|:---------------------------------------------|-------------:|---------------------:|:---------------------|-------------------:|--------------------:|
| DAFiR_Full_vs_Semantic_k4 |            50 |           5 |          4 |            0.414000 |          0.390900 |   0.023100 | [0.023099999999999954, 0.023099999999999954] |     0.000000 |            74.000000 | [74.0, 74.0]         |         459.000000 |            0.000000 |

Source: [results/week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv](../../week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv).

## 6. Figures

No saved figure is used for this part; no visualization is fabricated.

## 7. Statistical analysis

No formal inferential test was found for this comparison.

## 8. Interpretation

### Supported

The surviving final-comparison bootstrap is smoke-labelled. A full completed J DEV evaluation is not established.

### Not established

Point-estimate rankings do not establish statistical superiority. Implementation, qualification and retrospective oracle ceilings are not deployed or held-out quality results. Missing model/split/seed details are not established from available primary artifacts.

### Important caveats

A successful tiny bootstrap is software verification, not experimental significance.

## 9. Part conclusion

- The surviving final-comparison bootstrap is smoke-labelled. A full completed J DEV evaluation is not established.
- No additional method-selection decision is inferred beyond the recorded protocol/freeze artifacts.

# Week 8 — Integrated Analysis

## What changed during this week?

Investigate draft-aware selective evidence promotion, measured generator interaction and adaptation while separating real execution from archived/smoke prototypes. The parts above retain the distinction between executed outputs, implementation-only work, exploratory diagnostics and retrospective choices.

## Main supported findings

| Finding                                                                                                                                                                                             | Evidence                                                                                                            | Strength                                     |
|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|:---------------------------------------------|
| FIT has10476 candidate rows for1746 queries. Base F1 is 0.385631; the retrospective selective oracle is 0.485209.                                                                                   | Part A; results/week8/surfaces/promotion_surface_fit.jsonl                                                          | descriptive; see stated inferential evidence |
| Executed code-cell outputs record342 generated drafts and12 uncertainty features. The saved output also has degenerate logprob/length features (e.g. answer length mean0) despite completed drafts. | Part B; notebooks/week8/week8b_draft_diagnostics.ipynb; results/week8/drafts/draft_diagnostics_internal_val.jsonl   | PARTIALLY EXECUTED                           |
| U4-A FIT OOF oracle-gated headroom recovery is0.390104 versus DraftSemantic0.389770: an extremely small difference. Oracle-gated F1 must not be mistaken for unconditional promoted-answer quality. | Part C; results/week8/w8c_utility/fit_oof_comparison.csv; results/week8/w8c_utility/w8c_frozen_selector_config.json | descriptive; see stated inferential evidence |
| Masking measured10464/10476 FIT candidate rows; missing12 are not observed zeros. On INTERNAL-VAL, S3/S7 always-promote F1 is0.412572 versus base0.389061; U4-A is0.408329.                         | Part D; results/week8/w8d_attribution/full/features_fit.csv; results/week8/w8d_closeout/internal_val_metrics.csv    | descriptive; see stated inferential evidence |
| Separate FIT metrics and frozen-decision artifacts exist for the closeout. Always-promote and oracle-gated SHR columns are reported separately.                                                     | Part D-CLOSEOUT; results/week8/w8d_closeout/fit_metrics.csv; results/week8/w8d_closeout/fit_frozen_decisions.json   | descriptive; see stated inferential evidence |
| The 342-query INTERNAL-VAL table distinguishes no-promotion E0 F10.389061, always-promotion E1 0.412572, and gated E2 0.406122. Pipeline status is complete for this controller artifact set.       | Part E; results/week8/w8e_controller/internal_val_metrics.csv; results/week8/w8e_controller/controller_summary.json | descriptive; see stated inferential evidence |
| QCLA0/1/2 validation F1 is0.389061/0.393799/0.398994. QCLA2 doubles recorded cumulative prompt tokens; trainable parameters are about62.9M.                                                         | Part F; results/week8/w8f_qcla/full/comparison.csv; results/week8/w8f_qcla/full/summary.json                        | descriptive; see stated inferential evidence |
| No executed code outputs or completed full-run result artifact were found in the named G notebook/output paths.                                                                                     | Part G; notebooks/week8/week8g_outcome_weighted_sft.ipynb                                                           | IMPLEMENTED                                  |
| Only explicitly smoke-labelled full-system cost output was located; full-system completion is not established.                                                                                      | Part H; results/week8/tables/w8h_dafir1_cost_summary_smoke.csv                                                      | SMOKE-TESTED                                 |
| No completed result artifact was found for the named I full-run path.                                                                                                                               | Part I; notebooks/week8/week8i_ablations.ipynb                                                                      | IMPLEMENTED                                  |
| The surviving final-comparison bootstrap is smoke-labelled. A full completed J DEV evaluation is not established.                                                                                   | Part J; notebooks/week8/week8j_final_dev_evaluation.ipynb; results/week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv | SMOKE-TESTED                                 |

## Negative / null findings

Learned selector gain over a simple draft-semantic selector is tiny on FIT; missing masking and stale uncertainty features require care; full G/I/J evaluation is not established.

## What remained unresolved?

Same project development exposure; retrospective gates/oracles; missing continuous likelihood labels; archived invalidated runs; smoke/full ambiguity must not be collapsed.

## What logically motivated the next week?

Week9 freezes retrieval and separately tests evidence count, selector, representation, sentence suppression and revision.

## Offline reproducibility / source conflicts

- Recomputed promotion base/oracle metrics from10476 raw candidate rows; verified six candidates for each of1746 FIT queries.
- Feature table confirms10464 measured masking rows.
- G/H/I/J status is based on surviving code outputs and explicitly smoke-labelled files, not previous narrative completion claims.

Executed notebook coverage (zero-based cell indices):

| artifact                                           |   executed_output_cells | cell_indices                                      |
|:---------------------------------------------------|------------------------:|:--------------------------------------------------|
| notebooks/week8/week8c_utility_modeling.ipynb      |                      18 | 2,4,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37 |
| notebooks/week8/week8e_controller.ipynb            |                       5 | 1,3,5,7,9                                         |
| notebooks/week8/week8d_generator_interaction.ipynb |                       5 | 1,3,5,7,9                                         |
| notebooks/week8/week8b_draft_diagnostics.ipynb     |                       5 | 1,2,3,4,5                                         |
| notebooks/week8/week8j_final_dev_evaluation.ipynb  |                       0 |                                                   |
| notebooks/week8/week8f_qcla.ipynb                  |                       4 | 1,3,5,7                                           |
| notebooks/week8/week8g_outcome_weighted_sft.ipynb  |                       0 |                                                   |
| notebooks/week8/week8i_ablations.ipynb             |                       0 |                                                   |
| notebooks/week8/week8_master.ipynb                 |                       0 |                                                   |
| notebooks/week8/week8d_cached_closeout.ipynb       |                       3 | 1,2,3                                             |
| notebooks/week8/week8h_full_system.ipynb           |                       0 |                                                   |
| notebooks/week8/week8a_promotion_oracle.ipynb      |                       9 | 1,2,3,4,5,6,8,9,10                                |

# Primary Sources Used

- [configs/week8/w8_dafir.yaml](../../../configs/week8/w8_dafir.yaml)
- [notebooks/week8/week8_master.ipynb](../../../notebooks/week8/week8_master.ipynb)
- [notebooks/week8/week8a_promotion_oracle.ipynb](../../../notebooks/week8/week8a_promotion_oracle.ipynb)
- [notebooks/week8/week8b_draft_diagnostics.ipynb](../../../notebooks/week8/week8b_draft_diagnostics.ipynb)
- [notebooks/week8/week8c_utility_modeling.ipynb](../../../notebooks/week8/week8c_utility_modeling.ipynb)
- [notebooks/week8/week8d_cached_closeout.ipynb](../../../notebooks/week8/week8d_cached_closeout.ipynb)
- [notebooks/week8/week8d_generator_interaction.ipynb](../../../notebooks/week8/week8d_generator_interaction.ipynb)
- [notebooks/week8/week8e_controller.ipynb](../../../notebooks/week8/week8e_controller.ipynb)
- [notebooks/week8/week8f_qcla.ipynb](../../../notebooks/week8/week8f_qcla.ipynb)
- [notebooks/week8/week8g_outcome_weighted_sft.ipynb](../../../notebooks/week8/week8g_outcome_weighted_sft.ipynb)
- [notebooks/week8/week8h_full_system.ipynb](../../../notebooks/week8/week8h_full_system.ipynb)
- [notebooks/week8/week8i_ablations.ipynb](../../../notebooks/week8/week8i_ablations.ipynb)
- [notebooks/week8/week8j_final_dev_evaluation.ipynb](../../../notebooks/week8/week8j_final_dev_evaluation.ipynb)
- [results/week8/drafts/draft_diagnostics_internal_val.jsonl](../../week8/drafts/draft_diagnostics_internal_val.jsonl)
- [results/week8/surfaces/promotion_surface_fit.jsonl](../../week8/surfaces/promotion_surface_fit.jsonl)
- [results/week8/tables/w8h_dafir1_cost_summary_smoke.csv](../../week8/tables/w8h_dafir1_cost_summary_smoke.csv)
- [results/week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv](../../week8/tables/w8j_bootstrap_dafir_vs_k4_smoke.csv)
- [results/week8/w8c_utility/fit_oof_comparison.csv](../../week8/w8c_utility/fit_oof_comparison.csv)
- [results/week8/w8c_utility/w8c_frozen_selector_config.json](../../week8/w8c_utility/w8c_frozen_selector_config.json)
- [results/week8/w8d_attribution/full/features_fit.csv](../../week8/w8d_attribution/full/features_fit.csv)
- [results/week8/w8d_closeout/closeout_summary.json](../../week8/w8d_closeout/closeout_summary.json)
- [results/week8/w8d_closeout/fit_frozen_decisions.json](../../week8/w8d_closeout/fit_frozen_decisions.json)
- [results/week8/w8d_closeout/fit_metrics.csv](../../week8/w8d_closeout/fit_metrics.csv)
- [results/week8/w8d_closeout/internal_val_metrics.csv](../../week8/w8d_closeout/internal_val_metrics.csv)
- [results/week8/w8e_controller/controller_summary.json](../../week8/w8e_controller/controller_summary.json)
- [results/week8/w8e_controller/internal_val_metrics.csv](../../week8/w8e_controller/internal_val_metrics.csv)
- [results/week8/w8e_controller/pipeline_status.json](../../week8/w8e_controller/pipeline_status.json)
- [results/week8/w8f_qcla/full/comparison.csv](../../week8/w8f_qcla/full/comparison.csv)
- [results/week8/w8f_qcla/full/summary.json](../../week8/w8f_qcla/full/summary.json)

# Existing Markdown Used as Evidence

NONE

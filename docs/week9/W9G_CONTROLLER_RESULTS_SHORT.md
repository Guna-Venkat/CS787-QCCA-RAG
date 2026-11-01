# W9-G controller results — short interpretation

Reviewed saved Git working-tree artifacts only: **1,746 FIT queries, 656 papers**. No training or generation was rerun for this review.

## Can training finish in about one minute?

**Yes.** These are small logistic and gradient-boosted classifiers using **32 numeric features** and already-generated answers. The completed manifest records **zero LM calls**. Nested five-outer/four-inner paper folds plus final fits amount to 52 small estimator fits; there is no expensive answer-generation pass. Exact total elapsed time was not saved, so one minute is plausible rather than independently timed here.

## Answer quality and safety

| Policy | F1 | HELP vs D0 | HURT vs D0 |
| --- | ---: | ---: | ---: |
| D0: keep draft | 0.385631 | — | — |
| F0: always use fresh answer | 0.390024 | 21.65% | 20.90% |
| F2-A: evidence-anchored revision | 0.389616 | 2.18% | 0.86% |
| Logistic verifier — selected | **0.389443** | 7.45% | 6.24% |
| Gradient-boosted verifier | 0.389266 | 8.76% | 7.90% |

Logistic is slightly better than boosting by point F1 and causes less HURT. It accepts **321 proposals (18.38%)**: 130 helpful, 109 harmful and 82 neutral. It rejects **256/365 harmful proposals (70.14%)**, while retaining **130/378 helpful proposals (34.39%)**. Thus it distinguishes some useful changes, but still discards **65.61%** of F0's helpful changes. Of accepted proposals, **59.50% are non-HELP**, including neutral cases.

Compared with F2-A, logistic retains more useful F0 changes (34.39% versus 7.94%) but also accepts more harmful ones. F2-A still has slightly higher point F1 and much lower overall HURT. This is a tradeoff, not a clear dominance result.

## Statistical conclusion

Selected logistic paired paper-bootstrap differences:

- **vs D0:** +0.003812 F1; 95% CI **[−0.005019, +0.012492]**.
- **vs F0:** −0.000581 F1; 95% CI **[−0.010087, +0.008749]**.

Both intervals cross zero: **no established quality improvement over D0 or F0**. The high preflight oracle ceiling has not translated into a comparable learned gain.

The table uses nested paper-OOF predictions with thresholds selected inside each outer training partition. The final logistic deployment threshold is **0.70**, selected separately from pooled outer-OOF probabilities. Model/deployment selection remains FIT post-selection; upstream E3/F choices also used FIT. Independent confirmation is still needed.

## Cost, checks and decision

Every query still pays F0 generation: approximately **2,026 cumulative prompt tokens and two LM calls**, even when the verifier keeps D0. This is a quality/safety mechanism, not a generation-cost saving.

Artifact checksums, all 1,746 predictions per model, five paper folds and 40 inner-fold audit entries validated; no training/validation/outer-heldout paper overlap was found in those audits.

**Conclusion:** W9-G trained successfully and shows a useful preservation/retention tradeoff, but no demonstrated F1 breakthrough. Preserve the result as a conditional FIT-OOF experiment; do not infer deployable superiority from the oracle or point estimates.

Sources: `results/week9/w9g_action_verification/v1/controller_fit_oof/{summary.csv,bootstrap.csv,manifest.json,fold_audit.json,oof_predictions.csv,logistic_deployment.json}` and saved W9-F summaries.

# W8-D cached-feature closeout — 18 October 2026

The original extraction was repaired for question whitespace on 23 FIT and 5 INTERNAL-VAL queries. Canonical W7/W8-A generation strips questions; the initial masking code did not. Only those masking measurements were repeated. Original valid MiniLM semantic components and all W8-C model inputs were retained, and the closeout was rerun using the same fixed protocol. Superseded results and a repair ledger are preserved under `results/week8/archive_w8d_question_normalization/`.

No new attribution architecture search was performed. S5 ranks masking attribution (per token; equivalent to sum within one query), S6 ranks MIG, S7 adds alpha*attribution to DraftSemantic, and S8 also adds beta*MIG. Alpha=[0,.1,.3,1,3], beta=[0,.1,.3,1]. Each FIT outer fold tunes coefficients on training papers only. Final coefficients are frozen from all FIT before development confirmation.

U4-B uses the existing frozen encoder/fusion architecture plus masking sum, per-token masking, target count and availability. Duplicate A6 is excluded. Imputation and standardization are fitted within training papers; every optimizer step contains one query's six candidates. Seed42, five paper folds and three epochs are unchanged. A matched U4-A replay reproduced original OOF scores within 1.4e-7 before correction; unchanged BASE inputs/encoder features are verified before reusing this reference.

Frozen S7 coefficients: `{"alpha": 0.1, "beta": 0.0}`.

Frozen S8 coefficients: `{"alpha": 0.1, "beta": 0.0}`.

## FIT OOF

| Selector | Always-promote F1 | HELP | HURT | Mean regret | Oracle-gated SHR |
|---|---:|---:|---:|---:|---:|
| S3 | 0.405454 | 14.32% | 9.85% | 0.076444 | 38.98% |
| S5 | 0.396207 | 11.40% | 9.56% | 0.085691 | 29.28% |
| S6 | 0.403384 | 13.86% | 9.28% | 0.078514 | 36.93% |
| S7 | 0.405454 | 14.32% | 9.85% | 0.076444 | 38.98% |
| S8 | 0.400908 | 14.49% | 10.54% | 0.080990 | 37.25% |
| U4-A | 0.402770 | 14.55% | 10.31% | 0.079128 | 39.01% |
| U4-A-replay | 0.402770 | 14.55% | 10.31% | 0.079128 | 39.01% |
| U4-B | 0.403255 | 14.20% | 10.14% | 0.078643 | 39.54% |

## INTERNAL-VAL development confirmation

| Selector | Always-promote F1 | HELP | HURT | Mean regret | Oracle-gated SHR |
|---|---:|---:|---:|---:|---:|
| S3 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| S5 | 0.389016 | 11.11% | 11.11% | 0.091007 | 20.56% |
| S6 | 0.396327 | 10.82% | 9.94% | 0.083697 | 21.69% |
| S7 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| S8 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| U4-A | 0.408329 | 13.45% | 7.02% | 0.071695 | 28.67% |
| U4-B | 0.416716 | 13.45% | 6.43% | 0.063308 | 39.57% |

Oracle-gated SHR is explicitly a selector diagnostic. Always-promote F1 is the selection-only deployable quantity; no learned STOP performance is claimed here.

## Escalation-needed validation subset

True best candidate delta >0.01 identifies this retrospective diagnostic subset; it is never an online gate.

| Selector | Queries | HELP | HURT | Mean delta | Mean regret | Best selected (ties accepted) |
|---|---:|---:|---:|---:|---:|---:|
| S3 | 98 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| S5 | 98 | 38.78% | 17.35% | 0.065716 | 0.253981 | 28.57% |
| S6 | 98 | 37.76% | 16.33% | 0.069354 | 0.250343 | 26.53% |
| S7 | 98 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| S8 | 98 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| U4-A | 98 | 46.94% | 13.27% | 0.091646 | 0.228051 | 33.67% |
| U4-B | 98 | 46.94% | 9.18% | 0.126497 | 0.193200 | 34.69% |

## Paired paper bootstrap versus DraftSemantic

10,000 paper resamples, retaining every query in each sampled paper. Query-weighted estimates of fixed predictions; intervals do not include retraining variation or correct for all development experiments.

| Selector | F1 difference | 95% F1 interval | HURT difference | 95% HURT interval |
|---|---:|---:|---:|---:|
| S5 | -0.023556 | [-0.044765, -0.003072] | +1.17% | [-1.73%, +4.08%] |
| S6 | -0.016245 | [-0.034383, +0.001192] | +0.00% | [-2.64%, +2.67%] |
| S7 | +0.000000 | [+0.000000, +0.000000] | +0.00% | [+0.00%, +0.00%] |
| S8 | +0.000000 | [+0.000000, +0.000000] | +0.00% | [+0.00%, +0.00%] |
| U4-A | -0.004244 | [-0.027433, +0.018821] | -2.92% | [-6.19%, +0.31%] |
| U4-B | +0.004144 | [-0.017637, +0.025976] | -3.51% | [-6.85%, +0.00%] |

S7 changes 6 validation passage identities relative to DraftSemantic.

S8 changes 6 validation passage identities relative to DraftSemantic.

## Frozen decision

The confirmation candidate selected on FIT OOF was **S7**. The corrected closeout verdict is **no demonstrated attribution benefit in tested variants**. Online attribution retained: **False**.

The success rule was fixed before confirmation: F1 gain >=0.003 with a positive lower paired interval, or HURT reduction >=0.02 with a strictly negative upper interval and F1 noninferiority (lower interval >-0.003). Validation is not used to replace the FIT-selected candidate with whichever variant looks best afterward.

A lack of demonstrated gain in these tested variants is not proof that attribution can never help. Support for an existing answer while compressed and the counterfactual value of exposing a passage at full fidelity are distinct measurements. Negative masking means zeroing the SFR vector increased the likelihood of the existing draft, not that removing the passage improves correctness.

The original full extraction recorded about 8.8 hours of feature work. Targeted repair adds only the 28 affected queries. Any online attribution policy must account for seven SARA forward passes per nonempty draft; tiny unconfirmed gains do not establish operational value. Full results, candidate scores, fold coefficients, paired intervals and U4-B weights remain in this directory. Original W8-C checkpoints and outputs are unchanged. TEST remains untouched.

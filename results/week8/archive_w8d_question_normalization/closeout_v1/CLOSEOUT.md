# W8-D cached-feature closeout — 18 October 2026

Feature extraction was complete before this closeout. No new SARA attribution extraction or architecture search was performed. The new token-uncertainty backfill for W8-E is a separate experiment and does not change these results.

## Protocol

S5 selects maximum per-token masking attribution; empty drafts fall back to DraftSemantic. S6 selects maximum MIG. S7 combines DraftSemantic + alpha * per-token attribution. S8 additionally adds beta * MIG. Coefficients are selected in the training papers of each of five outer FIT folds; the held-out papers receive only those selected coefficients. Final coefficients are frozen from all FIT before INTERNAL-VAL confirmation.

Alpha grid: [0, 0.1, 0.3, 1, 3]. Beta grid: [0, 0.1, 0.3, 1]. No sign/feature/architecture search was performed. Both final S7 and S8 chose alpha=0.1, beta=0; they change six passage identities relative to DraftSemantic on INTERNAL-VAL, with identical aggregate F1, HELP and HURT metrics.

U4-B uses the existing frozen MiniLM cross-encoder plus the same numeric fusion architecture, adding masking sum, masking per-token mean, target token count and availability. Duplicate A6 was excluded. Median imputation and standardization are fitted only on training-fold papers. Frozen encoder representations are cached because they are label-independent; every optimizer step still contains one query's six candidates. Seed=42, five paper folds, three epochs, and the W8-C learning rate/loss are unchanged.

A matched U4-A replay reproduces the saved U4-A OOF utility scores to a maximum absolute difference of 1.35e-7 and identical downstream metrics. This validates the frozen-encoder shortcut rather than quietly changing the baseline.

## FIT OOF

| Selector | Always-promote F1 | HELP | HURT | Mean regret | Oracle-gated SHR |
|---|---:|---:|---:|---:|---:|
| S3 | 0.405454 | 14.32% | 9.85% | 0.076444 | 38.98% |
| S5 | 0.396296 | 11.45% | 9.62% | 0.085602 | 29.34% |
| S6 | 0.403384 | 13.86% | 9.28% | 0.078514 | 36.93% |
| S7 | 0.403947 | 14.20% | 9.91% | 0.077951 | 38.37% |
| S8 | 0.400908 | 14.49% | 10.54% | 0.080990 | 37.25% |
| U4-A | 0.402770 | 14.55% | 10.31% | 0.079128 | 39.01% |
| U4-A-replay | 0.402770 | 14.55% | 10.31% | 0.079128 | 39.01% |
| U4-B | 0.403290 | 14.26% | 10.14% | 0.078609 | 39.57% |

## INTERNAL-VAL: frozen variants, development confirmation

| Selector | Always-promote F1 | HELP | HURT | Mean regret | Oracle-gated SHR |
|---|---:|---:|---:|---:|---:|
| S3 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| S5 | 0.389282 | 11.40% | 11.11% | 0.090741 | 20.85% |
| S6 | 0.396327 | 10.82% | 9.94% | 0.083697 | 21.69% |
| S7 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| S8 | 0.412572 | 14.62% | 9.94% | 0.067451 | 44.76% |
| U4-A | 0.408329 | 13.45% | 7.02% | 0.071695 | 28.67% |
| U4-B | 0.416716 | 13.45% | 6.43% | 0.063308 | 39.57% |

Oracle-gated SHR is explicitly a selector diagnostic. Always-promote F1 is the deployable selection-only quantity; none of these results tests a learned STOP decision.

## Escalation-needed validation subset (98 queries)

This subset uses max true delta >0.01 only for retrospective analysis, not for choosing online actions.

| Selector | HELP | HURT | Mean delta | Mean regret | Tied-best selection |
|---|---:|---:|---:|---:|---:|
| S3 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| S5 | 39.80% | 17.35% | 0.066643 | 0.253053 | 28.57% |
| S6 | 37.76% | 16.33% | 0.069354 | 0.250343 | 26.53% |
| S7 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| S8 | 51.02% | 15.31% | 0.143096 | 0.176601 | 37.76% |
| U4-A | 46.94% | 13.27% | 0.091646 | 0.228051 | 33.67% |
| U4-B | 46.94% | 9.18% | 0.126497 | 0.193200 | 34.69% |

## Paired paper bootstrap versus DraftSemantic

10,000 resamples of papers, retaining all queries in each sampled paper. Estimates are query-weighted. Intervals are exploratory development-set intervals, not untouched TEST confirmation.

| Selector | F1 difference | 95% F1 interval | HURT-rate difference | 95% HURT interval |
|---|---:|---:|---:|---:|
| S5 | -0.023290 | [-0.044429, -0.002671] | +1.17% | [-1.73%, +4.08%] |
| S6 | -0.016245 | [-0.034383, +0.001192] | +0.00% | [-2.64%, +2.67%] |
| S7 | +0.000000 | [+0.000000, +0.000000] | +0.00% | [+0.00%, +0.00%] |
| S8 | +0.000000 | [+0.000000, +0.000000] | +0.00% | [+0.00%, +0.00%] |
| U4-A | -0.004244 | [-0.027433, +0.018821] | -2.92% | [-6.19%, +0.31%] |
| U4-B | +0.004144 | [-0.017637, +0.025976] | -3.51% | [-6.85%, +0.00%] |

## Decision

S7 was the attribution candidate selected on FIT OOF before validation was loaded for confirmation. It does not improve on DraftSemantic on validation. S5 and S6 are substantially worse in validation point estimates.

U4-B has an interesting exploratory point estimate: +0.004144 F1 and -3.51 percentage points HURT versus DraftSemantic. However its F1 interval is [-0.017637,+0.025976], and the HURT interval reaches zero. It also does not beat DraftSemantic on FIT OOF. This does not justify switching to U4-B after inspecting validation.

Before validation, the closeout required either an F1 gain >=0.003 with positive lower paired interval, or a HURT reduction >=0.02 with strictly negative upper interval and F1 noninferiority (lower interval >-0.003). No tested attribution variant establishes that combination. Therefore attribution is excluded from the online W8-E controller for this experiment. This is **no demonstrated added value in these tested variants**, not proof that attribution can never help.

The useful scientific distinction is that support for an existing answer in compressed form does not reliably establish the counterfactual value of exposing a passage at full fidelity. Negative masking means removal increased the likelihood of the existing draft; it is not a passage-removal quality recommendation.

The original extraction took about 8.8 GPU hours across these splits. The closeout does not demonstrate a quality/safety improvement large and reliable enough to justify including that online cost. Raw JSON summaries, paper-bootstrap results, candidate scores, fold predictions, and U4-B weights are retained in this directory. Original W8-C weights and metrics are unchanged. TEST remains untouched.

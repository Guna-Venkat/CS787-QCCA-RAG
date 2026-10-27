# Week 10 — Final root-cause study report

All planned R/C/G/D experiments are complete on **231 project DEV questions from 86 papers**. All three G readers are evaluated: Mistral, Qwen3-8B (nonthinking), and Llama3.1-8B. The frozen source/configuration and C/G question, paper, and exact evidence identities were validated. Historical partial-panel receipts are retained as historical records.

**Scope:** development diagnostics, not a fresh confirmation study. Project DEV maps to raw QASPER TRAIN. No QASPER TEST execution, policy training, faithful SeleCom evaluation, or external confirmation was added.

Validated experiment answer rows: **8922** — R=5544, C=1155, G=2079, D=144. Only Llama's 693 G answers were newly generated in this completion pass; previous completed answers were reused. Qualification calls are separate from these experiment rows.

## R — Retrieval foundation

The same local trained SARA/Mistral checkpoint was held fixed across FULL and SARA interfaces. BM25, MiniLM, and SFR were evaluated at k=2/4/6/8. FULL exposes k text chunks; SARA also exposes remaining top-10 latent evidence, so this is not an equal-information interface comparison.

| Retriever | FULL k2 | k4 | k6 | k8 | SARA k2 | k4 | k6 | k8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.299486 | 0.363272 | 0.383935 | 0.377042 | 0.297712 | 0.340516 | 0.375524 | 0.382535 |
| MiniLM | 0.347328 | 0.382600 | 0.389471 | 0.397290 | 0.322777 | 0.389217 | 0.389919 | 0.392034 |
| SFR | 0.339134 | 0.392866 | 0.419880 | 0.402261 | 0.310922 | 0.381233 | 0.421290 | 0.408454 |

| Retrieval at k6 | Any-gold recall | All-gold recall | Span coverage | Retrieved tokens |
|---|---:|---:|---:|---:|
| BM25 | 0.5045 | 0.3455 | 0.5092 | 1242.7 |
| MiniLM | 0.5000 | 0.3227 | 0.4956 | 1136.6 |
| SFR | 0.6182 | 0.4273 | 0.6200 | 1115.3 |

SFR is the strongest retrieval foundation in this surface. At k6, SFR−MiniLM F1 is +0.030409 FULL (95% paper CI [0.003019, 0.060660]) and +0.031372 SARA ([0.004948, 0.060979]). SFR−BM25 FULL also has positive CIs at k2 and k6. This supports a retrieval limitation; it does not establish that retrieval is the only bottleneck.

At SFR k6, SARA−FULL is only +0.001410 (95% CI [−0.017479, 0.021239]); the surface does not establish an incremental latent-interface advantage. k6 is the best point estimate in this development sweep, not a confirmed optimum. Recall averages exclude questions with no gold spans, while unmapped spans remain misses. These are paper-local full-corpus retrieval results, not global retrieval.

### Retrieval-stage oracle gaps

| Surface | Best fixed F1 | Exact oracle F1 | Gap | Top 10% share of oracle gain |
|---|---:|---:|---:|---:|
| R-BM25-full | 0.383935 | 0.457294 | 0.073359 | 72.1% |
| R-BM25-sara | 0.382535 | 0.460466 | 0.077931 | 75.6% |
| R-MiniLM-full | 0.397290 | 0.458078 | 0.060788 | 81.4% |
| R-MiniLM-sara | 0.392034 | 0.459923 | 0.067890 | 80.8% |
| R-SFR-full | 0.419880 | 0.478077 | 0.058197 | 72.4% |
| R-SFR-sara | 0.421290 | 0.475478 | 0.054188 | 71.0% |

Across the same k action set, the descriptive best-fixed/oracle gap falls from 0.07336 (BM25 FULL) to 0.05820 (SFR FULL), and from 0.07793 (BM25 SARA) to 0.05419 (SFR SARA). This is consistent with the retrieval foundation explaining part of the adaptive-budget gap. The residual SFR SARA gap is concentrated: about 71% of its gain comes from the top 10% of queries. Gap reductions themselves have not been assigned confidence intervals; the oracle is retrospective.

## C — Representation controls under frozen SFR

| Evidence representation | F1 | Mean prompt tokens | HELP vs none | HURT vs none |
|---|---:|---:|---:|---:|
| no_evidence | 0.154569 | 70.8 | 0.000 | 0.000 |
| full | 0.407095 | 1957.6 | 0.576 | 0.065 |
| sara_k4 | 0.381233 | 863.6 | 0.558 | 0.095 |
| lead256 | 0.321490 | 339.0 | 0.450 | 0.130 |
| qcs256 | 0.344075 | 339.2 | 0.450 | 0.113 |

FULL−QCS256 = +0.063020 (95% CI [0.019431, 0.109377]); FULL−SARA = +0.025862 (CI [−0.012674, 0.065663]); QCS−LEAD = +0.022585 (CI [−0.018719, 0.063554]). QCS reduces prompt burden, but the matched 256-token lexical controls do not establish a reliable query-conditioning gain. QCS is an exact-source textual surrogate, not a faithful SeleCom reproduction. The adapted C checkpoint differs from the public G Mistral reader; do not attribute cross-phase differences solely to compression or reader strength.

## G — All three readers on identical evidence

| Reader | Evidence | F1 | EM | ΔF1 vs same-reader none | HELP | HURT | Prompt tokens | Generation seconds |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral | no_evidence | 0.066586 | 0.000000 | +0.000000 | 0.000 | 0.000 | 41.8 | 0.666 |
| Mistral | full | 0.235322 | 0.004329 | +0.168735 | 0.619 | 0.130 | 1928.6 | 2.222 |
| Mistral | qcs256 | 0.189431 | 0.004329 | +0.122845 | 0.494 | 0.121 | 310.2 | 1.000 |
| Qwen | no_evidence | 0.073060 | 0.000000 | +0.000000 | 0.000 | 0.000 | 42.1 | 0.612 |
| Qwen | full | 0.273760 | 0.000000 | +0.200700 | 0.662 | 0.048 | 1754.2 | 1.243 |
| Qwen | qcs256 | 0.211520 | 0.008658 | +0.138459 | 0.485 | 0.113 | 272.8 | 0.731 |
| Llama | no_evidence | 0.068372 | 0.000000 | +0.000000 | 0.000 | 0.000 | 65.1 | 0.594 |
| Llama | full | 0.289647 | 0.034632 | +0.221275 | 0.662 | 0.087 | 1750.5 | 1.417 |
| Llama | qcs256 | 0.197178 | 0.025974 | +0.128806 | 0.476 | 0.121 | 292.5 | 0.673 |

HELP/HURT use paired F1 differences greater than +0.01 / less than −0.01 against each reader’s own no-evidence answer. Token counts use each reader’s native tokenizer/template, so they are workload measures, not a shared evidence-length measure. Generation timing is measured; full end-to-end latency is not measured. Each answer uses one LM call.

Generation-cap hits under the frozen 128-token budget: Mistral/no_evidence=0/231; Mistral/full=6/231; Mistral/qcs256=0/231; Qwen/no_evidence=0/231; Qwen/full=0/231; Qwen/qcs256=0/231; Llama/no_evidence=0/231; Llama/full=0/231; Llama/qcs256=0/231. Reader conclusions are conditional on this prompt/decoding budget, not universal model-capability rankings.

### Paired paper-cluster intervals

10,000 resamples, seed42, paper clusters; intervals are exploratory and unadjusted for multiple comparisons.

| Comparison (left minus right) | ΔF1 | 95% CI |
|---|---:|---|
| Mistral:full-no_evidence | +0.168735 | [+0.135667, +0.202014] |
| Mistral:qcs256-no_evidence | +0.122845 | [+0.092168, +0.154847] |
| Mistral:full-qcs256 | +0.045890 | [+0.020331, +0.071664] |
| Qwen:full-no_evidence | +0.200700 | [+0.167788, +0.234301] |
| Qwen:qcs256-no_evidence | +0.138459 | [+0.106183, +0.172358] |
| Qwen:full-qcs256 | +0.062241 | [+0.037946, +0.087982] |
| Llama:full-no_evidence | +0.221275 | [+0.183532, +0.259370] |
| Llama:qcs256-no_evidence | +0.128806 | [+0.093455, +0.166944] |
| Llama:full-qcs256 | +0.092469 | [+0.061442, +0.123764] |
| Mistral-Qwen:no_evidence | -0.006474 | [-0.018095, +0.004175] |
| Mistral-Qwen:full | -0.038439 | [-0.065226, -0.012804] |
| Mistral-Qwen:qcs256 | -0.022088 | [-0.039533, -0.004086] |
| (Mistral utility)-(Qwen utility):full | -0.031965 | [-0.059549, -0.005235] |
| (Mistral utility)-(Qwen utility):qcs256 | -0.015614 | [-0.037530, +0.007231] |
| Mistral-Llama:no_evidence | -0.001786 | [-0.013190, +0.009979] |
| Mistral-Llama:full | -0.054325 | [-0.080271, -0.029069] |
| Mistral-Llama:qcs256 | -0.007747 | [-0.029301, +0.011438] |
| (Mistral utility)-(Llama utility):full | -0.052540 | [-0.081045, -0.024391] |
| (Mistral utility)-(Llama utility):qcs256 | -0.005961 | [-0.031343, +0.017534] |
| Qwen-Llama:no_evidence | +0.004688 | [-0.007081, +0.017451] |
| Qwen-Llama:full | -0.015887 | [-0.037168, +0.005383] |
| Qwen-Llama:qcs256 | +0.014342 | [-0.008152, +0.036058] |
| (Qwen utility)-(Llama utility):full | -0.020575 | [-0.044262, +0.002492] |
| (Qwen utility)-(Llama utility):qcs256 | +0.009653 | [-0.015973, +0.034282] |

### Retrospective reader action oracles

| Reader | Best fixed F1 | Exact oracle F1 | Epsilon oracle F1 | Headroom |
|---|---:|---:|---:|---:|
| Mistral | 0.235322 | 0.283792 | 0.283715 | 0.048471 |
| Qwen | 0.273760 | 0.310372 | 0.310213 | 0.036612 |
| Llama | 0.289647 | 0.337771 | 0.337771 | 0.048124 |

These oracles retrospectively select among none/full/QCS using observed gold F1. They are not deployable controllers or evidence that a controller can recover the gap. R, C, and G use different action spaces; their oracle gaps are not directly comparable measures of controller difficulty.

## D — Action-label stability and annotation diagnostics

On the prespecified 24-query Mistral subset, 20/24 questions (83.3%) retain the same HELP/NEUTRAL/HURT label across all three sampling seeds. Mean modal label agreement is 0.944; mean delta-F1 variance is 0.005676; mean EM-label agreement is 1.000.

This is a small stochastic sensitivity diagnostic (temperature0.7/top-p0.9), not a test that greedy outputs change across runs, nor proof that F1 labels are semantically wrong. Primary reader decoding is greedy. All 72 sampled EM-delta labels are NEUTRAL; perfect EM-label stability is therefore degenerate and provides no useful evidence of reliable intervention labels.

Annotation inventory: 7/231 questions have multiple annotations; 11 have no gold evidence span. Semantic correctness and metric mismatch require a preregistered human audit and remain unmeasured.

| Sampling seed | Best fixed F1 | Exact oracle F1 | Epsilon oracle F1 |
|---|---:|---:|---:|
| 42 | 0.260841 | 0.260841 | 0.260841 |
| 43 | 0.278202 | 0.288450 | 0.288450 |
| 44 | 0.284169 | 0.284169 | 0.284169 |

## What Week 10 establishes

1. **Retrieval quality is a supported contributor.** SFR improves the matched retrieval/answering foundation and descriptively narrows the adaptive-k oracle gap. It should be retained for the next controlled study.
2. **The latent interface is not yet a demonstrated advantage.** The best local point estimate is SFR/SARA k6 (0.421290), but SFR/FULL k6 (0.419880) is essentially tied; their paired interval spans zero. Preserve the simple FULL control.
3. **Aggressive text suppression has a consistent quality cost.** FULL beats QCS256 with positive paper-cluster intervals for all three G readers. This supports a budget/representation limitation, not a claim that every learned or latent compressor fails.
4. **Reader utilization also matters.** Qwen and Llama improve full-evidence F1 and net evidence utility over public Mistral with positive paired intervals. Llama has the highest full-evidence point estimate (0.289647), while Qwen has the highest QCS256 point estimate (0.211520). Qwen-versus-Llama intervals include zero for both representations, so neither is a statistically established universal winner.
5. **Noise is a limitation, not an established dominant explanation.** Four of 24 sampled questions change their F1-derived intervention label across seeds. EM-label stability is degenerate: every sampled EM-delta label is NEUTRAL. This small diagnostic neither proves semantic metric correctness nor shows that the remaining oracle gap is mostly randomness.
6. **A deployable adaptive controller remains unproven.** Residual oracle gain is concentrated in relatively few questions, and no learned controller was trained here. A retrospective ceiling is not a controller result.

## Recommended next step

Freeze SFR as the retrieval foundation and retain SFR/FULL k6 plus the matched SFR/SARA k6 as controlled candidates with the existing trained local reader. The present G experiment supports reader effects within its public-reader setup; it does not isolate training/prompt differences between public G and adapted C/R checkpoints. Preregister a genuinely fresh confirmation comparison before investing in a new adaptive policy. If compression remains the research target, test a faithful trained compressor against FULL at explicit matched budgets; QCS256 alone does not settle that question. These are recommendations, not additional experiments launched by this report.

## Completion and provenance

Execution freeze: `6d603f7220cf6545e64a32227fad18497ea7d56dec8013925d08ab503fbd0488`. All reader model revisions remain as preregistered. C/G exact evidence equality, per-row checksums, expected query/action membership, paper IDs, completion receipts, and reader qualifications are checked by this report.

Scientific source/config changes: none. Original R/C results and earlier partial-panel/D provenance are preserved. `full_study_summary.json` is the current complete-panel receipt; earlier `partial_panel_status.json` describes the historical Llama deferral.

## Remaining research limits

- Fresh external confirmation is not complete; DEV is not a blind evaluation set.
- W9 FIT and W10 DEV scores have different cohorts/protocols; their raw F1 values are not a paired cross-week improvement estimate.
- Faithful SeleCom and a learned textual compressor were not evaluated.
- No semantic-correctness audit or evidence-entailment evaluation was performed.
- The D subset is small and Mistral-only; do not generalize its rate to the whole dataset or other readers.
- Oracle headroom does not establish a learnable adaptive policy; policy research remains disabled.

# W9-E complete results analysis, including E3

**Scope:** completed FIT results reviewed on 25 October 2026, branch `gagan`. This report covers every completed E policy and the six-value E3 surface, with D0–D3 as frozen references. No model generations were started for this review; existing result artifacts were read without modification.

## Main finding

**E3-k3 is the strongest W9-E point estimate: F1 = 0.390024.** It exceeds no intervention (D0 = 0.385631) by **0.004393 F1**, or **0.439 percentage points**, and raw two-chunk intervention (D1 = 0.385004) by **0.005020 F1**. However, both paired confidence intervals include zero. E3 is a promising compact representation diagnostic, not an established improvement over D0.

The original structured extraction policies did not produce a convincing improvement. E0 worsened F1; E1-Q, E1-QD and E2 stayed close to D0 largely because invalid or unsupported extraction caused them to retain the original answer on most queries. E3 removed that extraction-generation failure mode: every query produced source-validated sentence evidence, with zero extraction-LM calls.

**The primary frozen policy remains K_E=0.** The separate E3 fixed-k artifact selects k=3 for the FIT diagnostic. The dynamic GO result is an oracle-headroom screen, not a trained or evaluated controller.

## 1. What was held fixed

- **1,746 FIT queries across 656 papers.** This is not a DEV/INTERNAL-VAL result.
- Retrieval: **FIXED-5 / SEMANTIC / width 20**.
- Coarse selector: **DraftSemantic**, using the saved requested **K_E=2** chunks and their content hashes. Queries with fewer actual chunks retain the saved source selection.
- Generator: the existing trained SARA Mistral setup, canonical 4-FULL/6-HEAVY context/latent inputs, the D/E regeneration prompt semantics (`revision_mode=False`), greedy decoding and a 128-token answer cap. No generator fine-tuning or W9-F mode comparison was introduced.
- E3 sentence ranking uses the question only. Gold answers, draft text, downstream F1 and HELP/HURT do not enter sentence scoring.
- D0–D3 and E0–E2 are loaded as frozen references, not regenerated for E3.

F1 and exact match (EM) below use the 0–1 scale. HELP and HURT are query fractions with F1 changes above +0.01 or below −0.01 relative to saved D0. More HELP than HURT does not guarantee a positive mean F1 change: magnitudes also matter.

## 2. Experiment definitions: what each condition did

| Experiment | Evidence transformation and purpose |
| --- | --- |
| D0 | Keep the initial answer; no extra evidence intervention. Primary fixed-policy reference. |
| D1 | Regenerate with the same two selected raw coarse chunks. Tests raw evidence injection. |
| D2 | Earlier query-only generative extraction from those chunks, then answer generation. Reference for free-form extraction. |
| D3 | Earlier draft-aware generative extraction, then answer generation. Tests whether the draft improves extraction. |
| E0 | Deterministic query-only BM25-ranked source sentence/prefix spans, packed within a 128-token whole-evidence budget. No extraction LM. This is the original E0, not semantic Top-k. |
| E1-Q | Query-only LM extraction into strict structured support/fact objects, with exact-source checks and a 128-token evidence budget. |
| E1-QD | The same structured extraction, conditioned on the question and initial draft; the draft is not treated as source authority. |
| E2 | Draft-aware minimal blueprint with support, constrained claims and type/relevance labels; the same grounding and budget checks apply. |
| E3-SentTopK | Deterministically split the frozen chunks; embed sentences and question; select the highest-cosine k sentences; restore source order; regenerate using only those exact sentences. Six fixed k values are separate policies. No extraction LM. |

For E1/E2, intermediate extraction generation is capped at 256 tokens. Exact support and conservative source-subspan fact checks reject malformed/unsupported objects; unusable extraction retains D0 instead of substituting raw chunks. These restrictions improve mechanical traceability but can reject most generated objects.

E3 changes evidence exposure while holding upstream chunks and downstream generation fixed. Compared with E0, it also changes ranking, whole-sentence selection, formatting and budget definition; an E3–E0 difference cannot be attributed solely to semantic embeddings.

## 3. Overall answer-quality results

| Policy | F1 | EM | ΔF1 vs D0 | HELP | HURT | Answer changed |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 0.385631 | 0.227950 | +0.000000 | 0.00% | 0.00% | 0.00% |
| D1 | 0.385004 | 0.193013 | -0.000627 | 32.13% | 28.18% | 76.29% |
| D2 | 0.368659 | 0.205040 | -0.016972 | 21.08% | 23.88% | 64.60% |
| D3 | 0.376936 | 0.211340 | -0.008695 | 18.67% | 20.10% | 55.33% |
| E0 | 0.366109 | 0.209049 | -0.019522 | 17.87% | 22.11% | 59.62% |
| E1-Q | 0.382523 | 0.223940 | -0.003108 | 1.03% | 1.49% | 4.01% |
| E1-QD | 0.382099 | 0.226231 | -0.003532 | 1.20% | 1.72% | 4.81% |
| E2 | 0.383591 | 0.225659 | -0.002040 | 2.06% | 2.00% | 6.13% |
| E3-k1 | 0.376361 | 0.218786 | -0.009271 | 17.07% | 18.96% | 55.21% |
| E3-k2 | 0.381658 | 0.221077 | -0.003974 | 19.53% | 20.05% | 58.88% |
| E3-k3 | 0.390024 | 0.226231 | +0.004393 | 21.65% | 20.90% | 61.34% |
| E3-k4 | 0.380518 | 0.215349 | -0.005113 | 22.05% | 22.45% | 63.40% |
| E3-k6 | 0.385909 | 0.217640 | +0.000278 | 24.23% | 22.97% | 66.21% |
| E3-k8 | 0.379924 | 0.215349 | -0.005707 | 24.17% | 24.40% | 67.35% |


### Interpretation of the original E policies

- **E0: F1 0.366109.** It is worse than both D0 and raw D1. Exact-source extraction and a small token budget alone do not ensure that the resulting context helps the generator.
- **E1-Q: F1 0.382523.** Structured query-only extraction does not beat D0; very few queries actually receive usable evidence.
- **E1-QD: F1 0.382099.** Draft conditioning does not improve the aggregate result over E1-Q. Their small difference is inconclusive statistically.
- **E2: F1 0.383591.** It has the highest point estimate among the original E0/E1/E2 conditions, but remains below D0 and frequently falls back to it. Its advantage over E1-QD is not established by the paired interval.

### Interpretation of E3

- **k=3 wins the measured surface.** It is the only policy within 0.003 F1 of the maximum: the eligibility threshold is 0.387024. Thus the lower-token tie-break does not change the selected k.
- **More sentences do not monotonically improve F1.** k=4 and k=8 fall below D0; k=6 is almost equal to it. More exposed evidence adds context without consistently improving answer quality.
- At k=3, HELP is 21.65% and HURT is 20.90%. The descriptive HELP/(HELP+HURT) ratio is about 50.9%, not a large separation.
- k=3 EM is 0.226231 versus D0's 0.227950. Its F1 point gain is therefore not an exact-match gain.
- All six k values have 1,746 rows. The completed surface contains **10,476 query-policy rows**, with no missing k/query combinations, no E3 fail-closed rows and **100% recorded source-support validation**.

## 4. Why the structured methods stayed near baseline

| Policy | Malformed output | Extraction fail-closed | Extraction cap hit | Mean / median / p90 evidence tokens | Accepted exact-source support |
| --- | --- | --- | --- | --- | --- |
| E0 | 0.00% | 0.00% | 0.00% | 105.62 / 106 / 128 | 100.00% |
| E1-Q | 63.12% | 93.36% | 27.21% | 3.78 / 0 / 0 | 100.00% |
| E1-QD | 67.98% | 92.38% | 23.77% | 4.39 / 0 / 0 | 100.00% |
| E2 | 44.39% | 87.23% | 28.41% | 9.22 / 0 / 65 | 100.00% |


The E1-Q/E1-QD/E2 extraction acceptance counts are **116 / 133 / 223 out of 1,746**, respectively. Their median evidence length is zero because most queries fail closed. Those very low mean evidence lengths are not evidence of successful, efficient compression.

On the accepted/intervened subsets, mean F1 changes versus each query's D0 are approximately **−0.04678 / −0.04637 / −0.01597**, respectively. These subsets differ between methods and are not randomized, so the conditional figures diagnose the remaining accepted interventions rather than support a fair cross-method ranking.

The evidence-quality table's E0 extraction failure rate is zero. The answer rows separately record **two E0 empty-answer generation fallbacks**, which retained the original draft. Thus extraction failure and downstream answer failure must not be conflated.

Accepted source support is 100% after filtering; proposed exact-support rates for E1-Q/E1-QD/E2 are only about **54.24% / 58.72% / 58.07%**. Filtering removes unsupported proposals but does not establish relevance, entailment, completeness or answer correctness. Malformed-output, cap-hit and support-failure categories can overlap and should not be summed.

## 5. E3 mechanism and evidence compression

The splitter is **`academic-offset-rules-v1`**, a deterministic rule set for punctuation/paragraph boundaries, academic abbreviations, initials and acronyms. It preserves exact source substrings and offsets; coarse chunk-edge fragments remain possible. It is not an LLM splitter.

The encoder is **`sentence-transformers/all-MiniLM-L6-v2`**, frozen snapshot **`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**. The established encoder uses token windows, mean pooling and L2 normalization. Ranking is descending query–sentence cosine, then document-order chunk rank, local sentence order and stable sentence ID. Selection pools both chunks. Presentation restores selected sentences to source order rather than cosine order.

The evaluated grid is **{1,2,3,4,6,8}**. If fewer sentences exist, actual k is recorded without duplication. E3 has **no 128-token evidence cap**: it tests sentence count, keeping full selected sentences. Therefore it is not a length-matched ablation against E0/E1/E2.

| Policy | Mean actual k | Evidence tokens mean / median / p90 | Mean retained raw-token fraction | Mean selected cosine | Minimum-selected cosine, mean | Next-sentence margin, mean |
| --- | --- | --- | --- | --- | --- | --- |
| E3-k1 | 1.000 | 29.5 / 26 / 49.0 | 1.52% | 0.537 | 0.537 | 0.058 |
| E3-k2 | 2.000 | 64.2 / 59 / 94.0 | 3.30% | 0.508 | 0.478 | 0.035 |
| E3-k3 | 3.000 | 98.3 / 92 / 139.0 | 5.07% | 0.486 | 0.443 | 0.025 |
| E3-k4 | 4.000 | 134.1 / 126 / 182.0 | 6.90% | 0.469 | 0.418 | 0.020 |
| E3-k6 | 5.999 | 203.8 / 193 / 268.0 | 10.50% | 0.443 | 0.383 | 0.015 |
| E3-k8 | 7.994 | 274.7 / 262 / 356.5 | 14.14% | 0.423 | 0.356 | 0.011 |


At k=3, the evidence averages **98.3 tokens** and retains **5.07%** of the raw chunk tokens on the mean per-query ratio, suppressing about **94.93%**. This is evidence length, not total prompt length: the original context remains in the answer prompt.

For k=3, 41.35% of queries select all sentences from one coarse chunk; the mean same-chunk pair fraction is 60.90%. The mean adjacent-sentence pair rate is 8.51%, mean pairwise cosine is 0.482, and the near-duplicate pair rate (cosine ≥0.95) is 0.34%. These diagnostics do not show rampant near-duplication as an obvious explanation of the result. They also do not establish that selected sentences contain sufficient answer evidence. No MMR was added.

Gold-sentence recall/density is not reported because a validated mapping from gold evidence into this exact sentence inventory was not prepared. Missing evaluation is not zero recall; cosine and exact source support are not substitutes for it.

## 6. Cost and latency

Token counts and call counts below are mean costs per query for executing a policy, including the saved initial draft conceptually. They are not sums over the six-policy research sweep. Final prompt tokens count the last used prompt; cumulative prompt/generated tokens include earlier LM stages. Latency columns average recorded, non-null stage measurements: E1-Q/E1-QD/E2 answer latency is conditional on their 116/133/223 answer calls, rather than averaged over all 1,746 queries. The original draft is cached, so end-to-end latency was not measured.

| Policy | Final prompt tokens | Cumulative prompt tokens | Cumulative generated tokens | LM calls | Extraction latency, s | Answer latency, s |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | 940.5 | 940.5 | 16.7 | 1.000 | — | — |
| D1 | 3041.4 | 3981.9 | 56.5 | 2.000 | — | 2.262 |
| D2 | 1476.5 | 5618.4 | 730.1 | 3.000 | 27.692 | 1.197 |
| D3 | 1508.7 | 5690.2 | 639.9 | 3.000 | 24.249 | 1.162 |
| E0 | 1093.1 | 2033.6 | 40.9 | 2.000 | 0.005 | 1.094 |
| E1-Q | 947.4 | 3365.1 | 167.6 | 2.066 | 6.015 | 0.890 |
| E1-QD | 948.5 | 3455.7 | 157.3 | 2.076 | 5.687 | 0.763 |
| E2 | 955.7 | 3590.9 | 175.2 | 2.128 | 6.363 | 0.737 |
| E3-k1 | 1017.0 | 1957.5 | 38.5 | 2.000 | 0.000 | 0.957 |
| E3-k2 | 1051.7 | 1992.2 | 39.2 | 2.000 | 0.000 | 0.988 |
| E3-k3 | 1085.8 | 2026.3 | 39.9 | 2.000 | 0.000 | 1.023 |
| E3-k4 | 1121.5 | 2062.1 | 39.9 | 2.000 | 0.000 | 1.030 |
| E3-k6 | 1191.2 | 2131.8 | 42.0 | 2.000 | 0.000 | 1.117 |
| E3-k8 | 1262.1 | 2202.7 | 43.0 | 2.000 | 0.000 | 1.168 |


**k=3 reduces final prompt tokens by about 64.3% and cumulative prompt tokens by 49.1% relative to D1.** It remains more expensive than D0: approximately 2,026 versus 941 cumulative prompt tokens and two rather than one conceptual LM calls. Compression is a substantial saving against raw intervention, not against doing nothing.

E1/E2 may show small final prompts because of fallback, but the extraction LM has already incurred cost. Their roughly 3,365–3,591 cumulative prompt tokens and 157–175 generated tokens make those near-baseline answers expensive. Earlier D2/D3 free-form extraction is more expensive still.

E3 makes **zero extraction-LM calls**. It has 65,966 unique sentence embeddings and 1,664 unique query-text embeddings; repeated question text explains why unique texts are fewer than query IDs. Recorded unique embedding origin computation is **234.35 s for sentences** and **12.85 s for queries**. Sentence embeddings are reusable preprocessing/index work; query embeddings remain conceptual online work even when cached.

Recorded E3 mean lookup/computation latency is about **0.148 s**, with ranking about **0.00158 s** and k=3 answer generation about **1.023 s**. These stages share cached work across the surface and are not a measured end-to-end latency. Origin computation, current-run computation and cache lookup are separate fields; do not multiply shared index costs by six or portray cached embeddings as conceptually free.

## 7. Statistical evidence

The saved comparisons use **10,000 paired paper-cluster bootstrap resamples, seed 42**, with queries equally weighted in the reported mean. All intervals below are 95% intervals on raw ΔF1.

| Original E comparison | ΔF1 | 95% CI |
| --- | --- | --- |
| E0 - D0 | -0.019522 | [-0.033040, -0.006043] |
| E0 - D1 | -0.018895 | [-0.035091, -0.002404] |
| E1-Q - D0 | -0.003108 | [-0.006651, +0.000308] |
| E1-QD - D0 | -0.003532 | [-0.007989, +0.000813] |
| E1-QD - E1-Q | -0.000424 | [-0.004698, +0.003732] |
| E1-QD - D3 | +0.005162 | [-0.007864, +0.018664] |
| E2 - D0 | -0.002040 | [-0.006545, +0.002245] |
| E2 - E1-QD | +0.001493 | [-0.003346, +0.006177] |


| E3 comparison | ΔF1 | 95% CI |
| --- | --- | --- |
| E3-k1 - D0 | -0.009271 | [-0.022229, +0.003307] |
| E3-k1 - D1 | -0.008644 | [-0.025509, +0.008277] |
| E3-k2 - D0 | -0.003974 | [-0.017109, +0.008815] |
| E3-k2 - D1 | -0.003347 | [-0.018465, +0.012269] |
| E3-k3 - D0 | +0.004393 | [-0.008467, +0.017454] |
| E3-k3 - D1 | +0.005020 | [-0.010208, +0.020848] |
| E3-k4 - D0 | -0.005113 | [-0.018656, +0.008573] |
| E3-k4 - D1 | -0.004486 | [-0.018943, +0.010112] |
| E3-k6 - D0 | +0.000278 | [-0.013954, +0.014583] |
| E3-k6 - D1 | +0.000905 | [-0.013409, +0.015910] |
| E3-k8 - D0 | -0.005707 | [-0.020568, +0.008933] |
| E3-k8 - D1 | -0.005080 | [-0.018700, +0.008935] |
| best-fixed - E2 | +0.006433 | [-0.006623, +0.019558] |


**Every E3 interval crosses zero**, including selected k=3 versus D0, D1 and E2. Original E0 is clearly worse than D0 and D1 under its saved intervals. For the other original E policies, the intervals do not establish a gain.

These are FIT analyses. k was selected on the same FIT surface, so selected-policy intervals are post-selection descriptive evidence, and the fixed-k tests are not multiplicity-adjusted. A crossing-zero interval proves neither superiority nor equivalence. No held-out confirmation has been performed here.

## 8. Oracle sentence-count analysis

These are retrospective choices using observed per-query F1, **not deployable policies**. Exact ties choose the smallest k. The epsilon oracle chooses the smallest k within **0.01 F1** of the per-query maximum.

- **Intervention-only:** chooses among {1,2,3,4,6,8}.
- **Full-action:** also permits k=0, importing D0 exactly.
- **Local window:** a supplementary intervention-only analysis around selected k=3 using evaluated values {1,2,3,4,6}. It invents no k=5 or k=7.

| Action set | Oracle | F1 | Mean k | Median k | Mean evidence tokens | Mean cumulative prompt tokens |
| --- | --- | --- | --- | --- | --- | --- |
| intervention only | exact | 0.488910 | 2.183 | 1 | 71.6 | 1999.6 |
| intervention only | epsilon | 0.488827 | 2.148 | 1 | 70.4 | 1998.4 |
| full action | exact | 0.515247 | 1.319 | 0 | 44.7 | 1365.0 |
| full action | epsilon | 0.515166 | 1.281 | 0 | 43.6 | 1357.1 |
| local window intervention only | exact | 0.477142 | 1.841 | 1 | 59.2 | 1987.2 |
| local window intervention only | epsilon | 0.477069 | 1.815 | 1 | 58.4 | 1986.3 |


| k | Intervention-only exact count | Intervention-only ε count (%) | Full-action exact count | Full-action ε count (%) |
| --- | --- | --- | --- | --- |
| 0 | — | — | 1074 | 1088 (62.31%) |
| 1 | 1120 | 1134 (64.95%) | 206 | 206 (11.80%) |
| 2 | 175 | 173 (9.91%) | 114 | 112 (6.41%) |
| 3 | 125 | 126 (7.22%) | 89 | 88 (5.04%) |
| 4 | 100 | 93 (5.33%) | 78 | 71 (4.07%) |
| 6 | 121 | 120 (6.87%) | 95 | 95 (5.44%) |
| 8 | 105 | 100 (5.73%) | 90 | 86 (4.93%) |


The full-action epsilon oracle keeps D0 on **1,088 queries (62.31%)**. This means D0 is within 0.01 of the best measured action on those queries, not that they intrinsically require no evidence. The intervention-only epsilon oracle chooses k=1 on 64.95% of queries, partly reflecting the smallest-k preference: **763/1,746 queries (43.70%) have identical F1 at all six E3 k values**.

The remaining oracle headroom is substantial, but selecting the favorable action requires information unavailable at inference. Reported oracle costs are the costs of chosen actions, excluding the cost of discovering them; they are not the cost of running the entire six-k surface or a demonstrated controller.

## 9. Dynamic-k GO / NO-GO

The saved screen reports **GO**, with **`controller_trained_or_run=false`**.

| Preregistered screen | Observed result | Outcome |
| --- | --- | --- |
| Full-action epsilon oracle improves over best fixed E3 by ≥0.005 F1 | 0.515166 − 0.390024 = **+0.125142** | Pass |
| Within 0.003 F1 of fixed best, with ≥15% saving in an allowed evidence/final/cumulative prompt cost | Evidence −55.70%; final prompt −7.73%; cumulative prompt −33.03%; oracle F1 higher | Pass through evidence/cumulative cost |
| Non-degenerate intervention-only heterogeneity plus HELP/HURT variation and gain | Only k=1 has ≥15% epsilon-oracle mass | Fail |

This supports researching **whether and how much to intervene**, with an explicit skip option. It does not establish that a count-only controller can learn the oracle choices. The full-action gain combines sentence-count adaptation and avoiding harmful interventions. Any future controller needs inference-available features, paper-separated out-of-fold training/evaluation and comparison with both frozen k=3 and D0. No controller was trained or run in this review.

## 10. What W9-E establishes

| Question | Supported answer |
| --- | --- |
| Did raw K=2 solve evidence utilization? | No established gain: D1 is approximately D0 by point estimate. |
| Did free-form extraction solve it? | No. D2/D3 point estimates are lower; they also add substantial LM cost. |
| Did strict structured grounding solve it? | No. E1/E2 reject most extraction attempts and remain near baseline mainly through fallback. |
| Is a deterministic source-span budget sufficient? | No: E0 is worse than D0 and D1 under its saved paired intervals. |
| Does semantic sentence suppression avoid the extraction mechanism failures? | Yes operationally: complete source-validated evidence surface, no extraction LM, no E3 fallback. |
| Does it improve answer quality? | k=3 has the strongest point estimate, but a reliable improvement over D0/D1/E2 is not established. |
| Does it reduce evidence burden? | Yes relative to raw D1: approximately 95% less exposed evidence and 49% fewer cumulative prompt tokens at k=3. |
| Is dynamic stopping proven? | No. Oracle headroom passes the research GO screen; no deployable controller result exists. |

**Decision:** retain D0/K_E=0 as the primary frozen result; retain **E3-k3** as the selected compact-evidence FIT diagnostic. Describe E3 as an operationally simpler, source-preserving alternative with promising point quality and substantial savings over raw intervention. Do not claim a statistically demonstrated improvement, semantic faithfulness, causal evidence discovery or general RAG success.

## 11. Git review, provenance and remaining limits

The working tree contains an executed W9-E notebook plus a new E3 package/config/tests/results and documentation additions. The large notebook diff includes saved execution output, so its line count does not represent the amount of algorithm change. E3 is appended separately; the E0/E1-Q/E1-QD/E2 names and results are retained. The E3 full manifest records `complete=true`, `full_fit=true`, `old_artifacts_unchanged=true`, `extraction_lm_calls=0`, and `dynamic_controller_run=false`.

This review independently checked row coverage, paper/query counts, the complete six-value grid, recorded source-support validation, failure counts and the oracle tie diagnostic from saved rows. The original four E policies have **6,984 rows** and the six E3 policies have **10,476 rows**. It did not independently regenerate embeddings or model answers. The report adds documentation only.

Older implementation notes describe a pre-run snapshot and still say the full E3 run was not started. That historical status is superseded by the completed manifest and this results report. A separate 64-token E configuration exists; no completed 64-token full result is claimed here. No gold-sentence evaluation, held-out confirmation, or learned adaptive stopping result is available.

Authoritative result artifacts:


- [Original E answer/cost summary](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/fit/answer_and_cost_summary.csv)

- [Original E extraction quality](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/fit/evidence_quality.csv)

- [Original E paired bootstrap](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/diagnostics/w9c_k2/grounded_faithfulness/v1/budget_128/fit/paired_paper_bootstrap.json)

- [E3 summary](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/summary.csv)

- [E3 paired bootstrap](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/bootstrap.csv)

- [E3 oracle summary](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/oracle_summary.json)

- [E3 oracle distributions](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/oracle_distribution.csv)

- [Dynamic screen](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/dynamic_go_no_go.json)

- [Embedding cost totals](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/embedding_costs.json)

- [E3 completed manifest](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/fit/manifest.json)

- [Frozen E3 fixed-k selection](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/frozen_e3_senttopk.json)

- [E3 protocol](/home/gunavenkat/Downloads/CS787-RAG-Project/results/week9/w9e3_senttopk/v1/protocol.json)

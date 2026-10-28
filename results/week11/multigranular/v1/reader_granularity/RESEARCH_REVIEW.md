# Week 11 D — reader × evidence granularity: full research review

**D is complete for the available text surface.** The mechanical audit passed all 16,650 checksummed rows. The experiment contains 150 unique development queries (50 per dataset), not 16,650 independent examples. QASPER has 40 paper clusters; HotpotQA and SciFact each have 50 query/claim clusters. Each query has 37 executable conditions per reader. The 1,800 configured token-compression rows remain pending LLMLingua-2 qualification.

## What was tested

Mistral, Qwen and Llama used the same frozen SFR/P0 top-20 candidate retrieval, passage/sentence/exact-clause proposition exposure, lexical/semantic/hybrid selectors, and budgets 256/512/1024/FULL. Every reader used exactly the reviewed C source selections, restored to source order. Numeric budgets enforce the maximum native-token count across all three readers. Realized token use can differ by tokenizer and by selected granularity. FULL is the frozen capped candidate context, not the entire paper or unlimited context.

Each public reader retained its pinned model/revision, FP16 precision, native chat template, greedy 128-token decoding limit and frozen answer instructions. Qwen thinking remained disabled. Gold answers/evidence were used only for evaluation. Same-reader no-evidence controls define HELP/HURT (QA epsilon 0.01; classification exact).

## Main findings

1. **There is no universal best reader/granularity combination in this pilot.** Qwen has the largest HotpotQA point-estimate peak; Llama has the largest QASPER peak with FULL context. These peaks are selected on the same 50 queries and do not establish independently confirmed superiority.

2. **Point estimates vary with reader, representation and budget, but a formal interaction is not established.** At semantic passage/1,024 tokens, QASPER Qwen and Llama are almost tied; Llama has a larger FULL-context point estimate. HotpotQA Qwen has a higher 1,024-token passage point than its own FULL point. Same-budget sentence exposure does not consistently improve over passages. All eight additional QA reader-by-granularity interaction intervals include zero.

3. **SciFact is heavily confounded by output-format compliance.** Mistral has 1,849/1,850 invalid strict-format outputs; Llama has 1,826/1,850; Qwen has zero. Strict scores remain valid measurements of the frozen response contract, but they cannot support a clean conclusion about relative claim reasoning or evidence utilization. Explanations following labels are scored invalid by the preregistered exact-label rule.

4. **The useful continuation is controlled characterization, not selecting a universal winner or training a controller.** Keep Qwen/SFR/P0 as the pre-existing primary control for F/G; run the prescribed E reader/compressor panel, and retain D as exploratory evidence of reader/context dependence.

## Baselines and best observed condition per reader

QA entries are answer F1; SciFact entries are strict classification accuracy. Each row has 50 queries. Best conditions are retrospective; different rows can use different evidence settings. These peaks are not matched-condition reader effects.

| dataset   | reader   |   baseline |     best |    delta | granularity   | selector   | budget   |     HELP |     HURT |   evidence_tokens | interpretation                    |
|:----------|:---------|-----------:|---------:|---------:|:--------------|:-----------|:---------|---------:|---------:|------------------:|:----------------------------------|
| HotpotQA  | Llama    |   0.094824 | 0.443019 | 0.348195 | PROPOSITION   | hybrid     | FULL     | 0.680000 | 0.040000 |       1295.200000 | retrospective pilot peak          |
| HotpotQA  | Mistral  |   0.100453 | 0.324281 | 0.223828 | PASSAGE       | semantic   | 512      | 0.580000 | 0.140000 |        439.180000 | retrospective pilot peak          |
| HotpotQA  | Qwen     |   0.150250 | 0.480272 | 0.330022 | PASSAGE       | semantic   | 1024     | 0.620000 | 0.060000 |        798.360000 | retrospective pilot peak          |
| QASPER    | Llama    |   0.071760 | 0.289146 | 0.217386 | SENTENCE      | lexical    | FULL     | 0.720000 | 0.120000 |       4079.320000 | retrospective pilot peak          |
| QASPER    | Mistral  |   0.059805 | 0.200750 | 0.140944 | SENTENCE      | lexical    | 1024     | 0.640000 | 0.140000 |       1002.280000 | retrospective pilot peak          |
| QASPER    | Qwen     |   0.091228 | 0.243652 | 0.152425 | PASSAGE       | hybrid     | FULL     | 0.740000 | 0.060000 |       4144.280000 | retrospective pilot peak          |
| SciFact   | Llama    |   0.000000 | 0.040000 | 0.040000 | PASSAGE       | semantic   | FULL     | 0.040000 | 0.000000 |       4819.100000 | format-confounded strict accuracy |
| SciFact   | Mistral  |   0.000000 | 0.000000 | 0.000000 | PASSAGE       | lexical    | 256      | 0.000000 | 0.000000 |         16.440000 | format-confounded strict accuracy |
| SciFact   | Qwen     |   0.360000 | 0.780000 | 0.420000 | PASSAGE       | hybrid     | 1024     | 0.500000 | 0.080000 |        630.720000 | retrospective pilot peak          |

## Matched-condition results

This table holds retriever, selector, granularity and requested budget fixed across readers. Identical source text does not imply identical native token counts. The reader-specific answer contract/template is part of the frozen reader interface.

| dataset   | reader   | evidence_granularity   | budget   |   task_score |     HELP |     HURT |   actual_evidence_tokens |
|:----------|:---------|:-----------------------|:---------|-------------:|---------:|---------:|-------------------------:|
| HotpotQA  | Llama    | PASSAGE                | 1024     |     0.424796 | 0.700000 | 0.040000 |               762.960000 |
| HotpotQA  | Mistral  | PASSAGE                | 1024     |     0.309067 | 0.560000 | 0.200000 |               903.240000 |
| HotpotQA  | Qwen     | PASSAGE                | 1024     |     0.480272 | 0.620000 | 0.060000 |               798.360000 |
| HotpotQA  | Llama    | PASSAGE                | FULL     |     0.429692 | 0.680000 | 0.060000 |              1293.580000 |
| HotpotQA  | Mistral  | PASSAGE                | FULL     |     0.320750 | 0.560000 | 0.220000 |              1527.300000 |
| HotpotQA  | Qwen     | PASSAGE                | FULL     |     0.431145 | 0.580000 | 0.060000 |              1351.060000 |
| HotpotQA  | Llama    | SENTENCE               | 1024     |     0.335331 | 0.620000 | 0.100000 |               830.500000 |
| HotpotQA  | Mistral  | SENTENCE               | 1024     |     0.288463 | 0.540000 | 0.200000 |              1001.600000 |
| HotpotQA  | Qwen     | SENTENCE               | 1024     |     0.472093 | 0.620000 | 0.040000 |               866.740000 |
| HotpotQA  | Llama    | SENTENCE               | FULL     |     0.426510 | 0.700000 | 0.040000 |              1293.560000 |
| HotpotQA  | Mistral  | SENTENCE               | FULL     |     0.313041 | 0.560000 | 0.240000 |              1568.640000 |
| HotpotQA  | Qwen     | SENTENCE               | FULL     |     0.405590 | 0.580000 | 0.060000 |              1351.020000 |
| QASPER    | Llama    | PASSAGE                | 1024     |     0.198886 | 0.580000 | 0.180000 |               759.700000 |
| QASPER    | Mistral  | PASSAGE                | 1024     |     0.192379 | 0.700000 | 0.100000 |               848.440000 |
| QASPER    | Qwen     | PASSAGE                | 1024     |     0.199183 | 0.600000 | 0.100000 |               773.900000 |
| QASPER    | Llama    | PASSAGE                | FULL     |     0.278155 | 0.680000 | 0.140000 |              4076.580000 |
| QASPER    | Mistral  | PASSAGE                | FULL     |     0.180881 | 0.680000 | 0.100000 |              4553.040000 |
| QASPER    | Qwen     | PASSAGE                | FULL     |     0.243652 | 0.740000 | 0.060000 |              4144.280000 |
| QASPER    | Llama    | SENTENCE               | 1024     |     0.209731 | 0.620000 | 0.120000 |               858.520000 |
| QASPER    | Mistral  | SENTENCE               | 1024     |     0.181724 | 0.660000 | 0.100000 |              1002.420000 |
| QASPER    | Qwen     | SENTENCE               | 1024     |     0.204816 | 0.560000 | 0.140000 |               872.960000 |
| QASPER    | Llama    | SENTENCE               | FULL     |     0.289146 | 0.720000 | 0.120000 |              4079.320000 |
| QASPER    | Mistral  | SENTENCE               | FULL     |     0.194480 | 0.660000 | 0.160000 |              4812.160000 |
| QASPER    | Qwen     | SENTENCE               | FULL     |     0.235650 | 0.700000 | 0.080000 |              4147.040000 |
| SciFact   | Llama    | PASSAGE                | 1024     |     0.000000 | 0.000000 | 0.000000 |               653.920000 |
| SciFact   | Mistral  | PASSAGE                | 1024     |     0.000000 | 0.000000 | 0.000000 |               800.320000 |
| SciFact   | Qwen     | PASSAGE                | 1024     |     0.680000 | 0.440000 | 0.120000 |               667.620000 |
| SciFact   | Llama    | PASSAGE                | FULL     |     0.040000 | 0.040000 | 0.000000 |              4819.100000 |
| SciFact   | Mistral  | PASSAGE                | FULL     |     0.000000 | 0.000000 | 0.000000 |              5917.040000 |
| SciFact   | Qwen     | PASSAGE                | FULL     |     0.660000 | 0.420000 | 0.120000 |              4936.440000 |
| SciFact   | Llama    | SENTENCE               | 1024     |     0.000000 | 0.000000 | 0.000000 |               805.480000 |
| SciFact   | Mistral  | SENTENCE               | 1024     |     0.000000 | 0.000000 | 0.000000 |              1000.180000 |
| SciFact   | Qwen     | SENTENCE               | 1024     |     0.620000 | 0.420000 | 0.160000 |               824.500000 |
| SciFact   | Llama    | SENTENCE               | FULL     |     0.020000 | 0.020000 | 0.000000 |              4826.640000 |
| SciFact   | Mistral  | SENTENCE               | FULL     |     0.000000 | 0.000000 | 0.000000 |              6043.260000 |
| SciFact   | Qwen     | SENTENCE               | FULL     |     0.680000 | 0.440000 | 0.120000 |              4943.660000 |

## Paired reader comparisons

The following additional analyses were specified after D completed and are labelled post-hoc exploratory. Every comparison pairs the same 50 queries; 10,000 resamples, seed 42; paper-cluster bootstrap for QASPER and query/claim bootstrap otherwise. Intervals are nominal, without multiple-comparison or winner-selection correction. A CI touching zero is not a clean positive result. SciFact intervals are about strict formatting-and-correctness performance, not isolated reasoning ability.

| dataset   | granularity   | budget   | reader_left   | reader_right   | family                         |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:--------------|:---------|:--------------|:---------------|:-------------------------------|----------:|----------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | PASSAGE       | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.171205 |  0.082889 |  0.262096 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.115729 |  0.015708 |  0.215728 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | 1024     | Llama         | Qwen           | additional_posthoc_exploratory | -0.055476 | -0.144145 |  0.031552 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.110395 |  0.020609 |  0.201671 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.108942 |  0.024269 |  0.194660 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | PASSAGE       | FULL     | Llama         | Qwen           | additional_posthoc_exploratory | -0.001453 | -0.074098 |  0.069387 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.183631 |  0.090885 |  0.277296 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.046868 | -0.057818 |  0.146451 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | 1024     | Llama         | Qwen           | additional_posthoc_exploratory | -0.136762 | -0.241214 | -0.037176 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.092549 |  0.006287 |  0.181631 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.113470 |  0.030477 |  0.197683 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | SENTENCE      | FULL     | Llama         | Qwen           | additional_posthoc_exploratory |  0.020921 | -0.042459 |  0.087536 |       10000 |     42 | query_or_claim_id |
| QASPER    | PASSAGE       | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.006805 | -0.031635 |  0.044873 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.006507 | -0.036294 |  0.057007 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | 1024     | Llama         | Qwen           | additional_posthoc_exploratory | -0.000297 | -0.048737 |  0.053130 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.062771 |  0.015145 |  0.115129 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.097274 |  0.040443 |  0.157083 |       10000 |     42 | paper_id          |
| QASPER    | PASSAGE       | FULL     | Llama         | Qwen           | additional_posthoc_exploratory |  0.034503 | -0.031313 |  0.098838 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.023093 | -0.006593 |  0.055368 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.028008 | -0.013750 |  0.072109 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | 1024     | Llama         | Qwen           | additional_posthoc_exploratory |  0.004915 | -0.049598 |  0.057236 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.041170 | -0.019502 |  0.104968 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.094666 |  0.044431 |  0.147285 |       10000 |     42 | paper_id          |
| QASPER    | SENTENCE      | FULL     | Llama         | Qwen           | additional_posthoc_exploratory |  0.053496 | -0.015459 |  0.124248 |       10000 |     42 | paper_id          |
| SciFact   | PASSAGE       | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.680000 |  0.540000 |  0.800000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | 1024     | Llama         | Qwen           | additional_posthoc_exploratory | -0.680000 | -0.800000 | -0.540000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.660000 |  0.520000 |  0.780000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.040000 |  0.000000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | PASSAGE       | FULL     | Llama         | Qwen           | additional_posthoc_exploratory | -0.620000 | -0.760000 | -0.480000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | 1024     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.620000 |  0.480000 |  0.760000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | 1024     | Llama         | Mistral        | additional_posthoc_exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | 1024     | Llama         | Qwen           | additional_posthoc_exploratory | -0.620000 | -0.760000 | -0.480000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | FULL     | Qwen          | Mistral        | additional_posthoc_exploratory |  0.680000 |  0.540000 |  0.800000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | FULL     | Llama         | Mistral        | additional_posthoc_exploratory |  0.020000 |  0.000000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | SENTENCE      | FULL     | Llama         | Qwen           | additional_posthoc_exploratory | -0.660000 | -0.800000 | -0.520000 |       10000 |     42 | query_or_claim_id |

For example, Llama minus Qwen on QASPER FULL passages is +0.034503 with CI [-0.031313, 0.098838]; on FULL sentences it is +0.053496 with CI [-0.015459, 0.124248]. Neither establishes superiority. On HotpotQA 1,024-token passages, Qwen minus Llama is +0.055476 with CI [-0.031552, 0.144145]; on 1,024-token sentences it is +0.136762 with nominal CI [0.037176, 0.241214]. The latter is an exploratory matched-condition signal, not a universal reader result.

## Does sentence exposure beat passages at the same requested budget?

The contrast is sentence minus passage with semantic selection. Actual token use is shown because requested-budget matching is not exact length matching. A positive point estimate alone is insufficient for superiority. FULL sentence/passage formatting and coverage can differ slightly even when nearly all source material is exposed.

| dataset   | reader   | budget   | contrast               |   sentence_tokens |   passage_tokens | family                         |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|:---------|:-----------------------|------------------:|-----------------:|:-------------------------------|----------:|----------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | Mistral  | 1024     | sentence_minus_passage |       1001.600000 |       903.240000 | additional_posthoc_exploratory | -0.020604 | -0.057401 |  0.012119 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  | FULL     | sentence_minus_passage |       1568.640000 |      1527.300000 | additional_posthoc_exploratory | -0.007710 | -0.022063 |  0.004242 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | 1024     | sentence_minus_passage |        866.740000 |       798.360000 | additional_posthoc_exploratory | -0.008179 | -0.073761 |  0.057367 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Qwen     | FULL     | sentence_minus_passage |       1351.020000 |      1351.060000 | additional_posthoc_exploratory | -0.025556 | -0.065569 |  0.004444 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | 1024     | sentence_minus_passage |        830.500000 |       762.960000 | additional_posthoc_exploratory | -0.089465 | -0.169546 | -0.020935 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | FULL     | sentence_minus_passage |       1293.560000 |      1293.580000 | additional_posthoc_exploratory | -0.003182 | -0.042000 |  0.031960 |       10000 |     42 | query_or_claim_id |
| QASPER    | Mistral  | 1024     | sentence_minus_passage |       1002.420000 |       848.440000 | additional_posthoc_exploratory | -0.010655 | -0.052807 |  0.030089 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  | FULL     | sentence_minus_passage |       4812.160000 |      4553.040000 | additional_posthoc_exploratory |  0.013599 | -0.011192 |  0.037830 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | 1024     | sentence_minus_passage |        872.960000 |       773.900000 | additional_posthoc_exploratory |  0.005633 | -0.035012 |  0.048284 |       10000 |     42 | paper_id          |
| QASPER    | Qwen     | FULL     | sentence_minus_passage |       4147.040000 |      4144.280000 | additional_posthoc_exploratory | -0.008002 | -0.032604 |  0.010195 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | 1024     | sentence_minus_passage |        858.520000 |       759.700000 | additional_posthoc_exploratory |  0.010846 | -0.061549 |  0.075662 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | FULL     | sentence_minus_passage |       4079.320000 |      4076.580000 | additional_posthoc_exploratory |  0.010991 | -0.007716 |  0.031742 |       10000 |     42 | paper_id          |
| SciFact   | Mistral  | 1024     | sentence_minus_passage |       1000.180000 |       800.320000 | additional_posthoc_exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Mistral  | FULL     | sentence_minus_passage |       6043.260000 |      5917.040000 | additional_posthoc_exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | 1024     | sentence_minus_passage |        824.500000 |       667.620000 | additional_posthoc_exploratory | -0.060000 | -0.180000 |  0.060000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Qwen     | FULL     | sentence_minus_passage |       4943.660000 |      4936.440000 | additional_posthoc_exploratory |  0.020000 | -0.040000 |  0.100000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | 1024     | sentence_minus_passage |        805.480000 |       653.920000 | additional_posthoc_exploratory |  0.000000 |  0.000000 |  0.000000 |       10000 |     42 | query_or_claim_id |
| SciFact   | Llama    | FULL     | sentence_minus_passage |       4826.640000 |      4819.100000 | additional_posthoc_exploratory | -0.020000 | -0.060000 |  0.000000 |       10000 |     42 | query_or_claim_id |

Llama HotpotQA sentence minus passage at 1,024 requested tokens is -0.089465, nominal CI [-0.169546, -0.020935], under this allocation policy. Qwen and Mistral intervals for the same contrast include zero. A negative within-reader interval versus an inconclusive interval in another reader does not itself prove an interaction; the direct difference-in-differences test is required.

## Reader × granularity interaction

These eight additional post-hoc QA contrasts compare the sentence-minus-passage effect for a reader against the same effect for Qwen. They are separate from the twelve explicitly configured H interaction contrasts. Interpret their nominal CIs as exploratory; do not treat descriptive factorial marginal averages as causal effects.

| dataset   | reader   | reference   | budget   | contrast                                            | family                         |     point |    ci_low |   ci_high |   resamples |   seed | cluster           |
|:----------|:---------|:------------|:---------|:----------------------------------------------------|:-------------------------------|----------:|----------:|----------:|------------:|-------:|:------------------|
| HotpotQA  | Mistral  | Qwen        | 1024     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory | -0.012425 | -0.081771 |  0.053866 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Mistral  | Qwen        | FULL     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory |  0.017846 | -0.017028 |  0.061327 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | Qwen        | 1024     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory | -0.081286 | -0.172899 |  0.004410 |       10000 |     42 | query_or_claim_id |
| HotpotQA  | Llama    | Qwen        | FULL     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory |  0.022373 | -0.024150 |  0.073229 |       10000 |     42 | query_or_claim_id |
| QASPER    | Mistral  | Qwen        | 1024     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory | -0.016288 | -0.058948 |  0.024116 |       10000 |     42 | paper_id          |
| QASPER    | Mistral  | Qwen        | FULL     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory |  0.021602 | -0.006623 |  0.049087 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | Qwen        | 1024     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory |  0.005213 | -0.057551 |  0.058055 |       10000 |     42 | paper_id          |
| QASPER    | Llama    | Qwen        | FULL     | (sentence-passage)_reader - (sentence-passage)_Qwen | additional_posthoc_exploratory |  0.018994 | -0.004856 |  0.043399 |       10000 |     42 | paper_id          |

All eight direct interaction CIs include zero. The observed patterns motivate further controlled study but do not justify a confirmed reader × granularity interaction claim on this cohort.

## Flags and suspicious records

| dataset   | reader   |   empty_evidence_rows |   generation_cap_rows |   invalid_label_rows |
|:----------|:---------|----------------------:|----------------------:|---------------------:|
| HotpotQA  | Llama    |                     2 |                     0 |                    0 |
| HotpotQA  | Mistral  |                     2 |                     4 |                    0 |
| HotpotQA  | Qwen     |                     2 |                     0 |                    0 |
| QASPER    | Llama    |                   144 |                    12 |                    0 |
| QASPER    | Mistral  |                   144 |                    58 |                    0 |
| QASPER    | Qwen     |                   144 |                     0 |                    0 |
| SciFact   | Llama    |                   162 |                   649 |                 1826 |
| SciFact   | Mistral  |                   162 |                   264 |                 1849 |
| SciFact   | Qwen     |                   162 |                     0 |                    0 |

Counts above are logical condition rows, not independent queries. There are 924 empty-evidence rows: 308 shared query/condition selections repeated across three readers. This follows the frozen strict-prefix allocation rule: an oversized first unit stops selection, so coarse passages can expose nothing at small budgets. Preserve this limitation instead of changing the allocation after seeing outcomes.

There are 987 generation-cap rows: Mistral 326, Llama 661, Qwen 0. Llama accounts for 649 SciFact cap hits; Mistral accounts for 264. This supports a response-format/verbosity concern. Cap hits can truncate otherwise meaningful answers; they were not silently retried with a larger limit. QA token F1 can also penalize explanatory verbosity, so reader F1 differences are not pure knowledge measurements.

Seven QASPER queries contain incompletely mapped evidence annotations (four fully unmapped, three partially mapped). Their 777 flagged logical rows are 7 queries × 37 conditions × 3 readers. Annotation gaps constrain evidence-recall interpretation; answer-quality scoring is retained. SciFact has 25 no-evidence-label claims with null retrieval recall rather than fabricated zero recall; classification still uses all 50 claims.

Illustrative saved records follow. Full query/action flags remain in `flagged_queries.csv`; excerpts do not replace the original predictions.

| dataset   | reader   | flag         |   query_id | action_id            | prediction_excerpt                                                                                                                                                                                                           |   score |   evidence_tokens |   generated_tokens |
|:----------|:---------|:-------------|-----------:|:---------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|------------------:|-------------------:|
| SciFact   | Mistral  | flag_invalid |       1024 | 0201e568dc562f50954e | NOT_ENOUGH_INFO. The claim refers to the frequency of recurrent mutations within CTCF anchor sites adjacent to oncogenes, but no specific data or studies are provided as evidence in the context given.                     |       0 |                 0 |                 47 |
| SciFact   | Llama    | flag_invalid |       1024 | 02a6f3329796db882b78 | SUPPORT  Reasoning: The claim is supported by the evidence in the text, which states that "The CTCF anchor regions we identified are a frequent site of mutations in cancer cells, underscoring their functional importance  |       0 |               221 |                 85 |
| QASPER    | Mistral  | flag_cap     |       1013 | 6a1e86c7a6280b2663bf | Previous applications of neural networks for this task often used unidimensional Convolutional Neural Networks (CNNs) for text embeddings and Recurrent Neural Networks (RNNs) for sequential data. However, this hybrid mod |       0 |               822 |                128 |
| QASPER    | Llama    | flag_empty   |       1003 | 97b63644eda906106fc8 | The authors provide data from the IEMOCAP dataset and the AMI corpus to support their assertion.                                                                                                                             |       0 |                 0 |                 22 |

## Separate post-hoc label-compliance diagnostic

This diagnostic accepts only an unambiguous exact leading label followed by an allowed boundary/punctuation. It does not search explanations, paraphrase labels or modify primary scoring. It was added after inspecting format failures. The values below help identify a formatting confound; they are not a replacement accuracy leaderboard, confirmatory claim, or inference-time feature.

| reader   | evidence_granularity   | selector   |   budget |   strict_accuracy |   strict_format_rate |   leading_label_rate |   leading_label_diagnostic_accuracy |   cap_hit_rate |
|:---------|:-----------------------|:-----------|---------:|------------------:|---------------------:|---------------------:|------------------------------------:|---------------:|
| Llama    | NONE                   | none       |        0 |          0.000000 |             0.000000 |             0.300000 |                            0.100000 |       0.540000 |
| Llama    | PASSAGE                | semantic   |     1024 |          0.000000 |             0.000000 |             1.000000 |                            0.420000 |       0.260000 |
| Llama    | SENTENCE               | semantic   |      256 |          0.000000 |             0.000000 |             1.000000 |                            0.440000 |       0.100000 |
| Mistral  | PASSAGE                | semantic   |     1024 |          0.000000 |             0.000000 |             0.980000 |                            0.680000 |       0.040000 |
| Mistral  | SENTENCE               | semantic   |      256 |          0.000000 |             0.000000 |             1.000000 |                            0.700000 |       0.000000 |
| Mistral  | NONE                   | none       |        0 |          0.000000 |             0.000000 |             0.920000 |                            0.440000 |       0.160000 |
| Qwen     | NONE                   | none       |        0 |          0.360000 |             1.000000 |             1.000000 |                            0.360000 |       0.000000 |
| Qwen     | SENTENCE               | semantic   |      256 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| Qwen     | PASSAGE                | semantic   |     1024 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |

## Retrospective oracle headroom

| dataset   | reader      | best_fixed           |   best_fixed_score |   exact_oracle |   epsilon_oracle |   oracle_headroom |   epsilon |
|:----------|:------------|:---------------------|-------------------:|---------------:|-----------------:|------------------:|----------:|
| HotpotQA  | ALL_READERS | 39d891f9fed0db18d8c2 |           0.480272 |       0.708511 |         0.708511 |          0.228239 |  0.010000 |
| HotpotQA  | Llama       | 12dd48b33d6d3fb8d217 |           0.443019 |       0.586837 |         0.586837 |          0.143818 |  0.010000 |
| HotpotQA  | Mistral     | f70536eca71d35e6b7a0 |           0.324281 |       0.436404 |         0.436257 |          0.112123 |  0.010000 |
| HotpotQA  | Qwen        | 39d891f9fed0db18d8c2 |           0.480272 |       0.605618 |         0.605618 |          0.125346 |  0.010000 |
| QASPER    | ALL_READERS | 2554cee82fe23d813c59 |           0.289146 |       0.473970 |         0.473557 |          0.184824 |  0.010000 |
| QASPER    | Llama       | 2554cee82fe23d813c59 |           0.289146 |       0.406540 |         0.406386 |          0.117394 |  0.010000 |
| QASPER    | Mistral     | e177fe617c51719a0d13 |           0.200750 |       0.361895 |         0.361810 |          0.161146 |  0.010000 |
| QASPER    | Qwen        | 05b2cedae417707e4aeb |           0.243652 |       0.359169 |         0.359083 |          0.115517 |  0.010000 |
| SciFact   | ALL_READERS | ee2068a1bcf2d92d6dd1 |           0.780000 |       0.940000 |         0.940000 |          0.160000 |  0.000000 |
| SciFact   | Llama       | 56b4550963a50e30655b |           0.040000 |       0.040000 |         0.040000 |          0.000000 |  0.000000 |
| SciFact   | Mistral     | 0201e568dc562f50954e |           0.000000 |       0.000000 |         0.000000 |          0.000000 |  0.000000 |
| SciFact   | Qwen        | ee2068a1bcf2d92d6dd1 |           0.780000 |       0.940000 |         0.940000 |          0.160000 |  0.000000 |

ALL_READERS lets an oracle choose both reader and evidence condition using the observed answers. That is an optimistic retrospective ceiling and ignores the serving burden of swapping reader models. It does not show that a dynamic controller works. The strict SciFact oracle inherits the label-compliance confound. No controller was trained.

## Cost and reproducibility

D recorded 7,012 physical generation calls: 3,506 Mistral and 3,506 Llama. All 5,550 Qwen results exactly reuse C predictions and scores. Other identical-evidence cases also reuse cached answers. Logical LM calls remain per-policy costs; cached answers are not independent repetitions. C computation and operational qualification calls are separate from D physical-call counters.

Reader process isolation changed operational placement only. Pinned weights, canonical generation, prompts, evidence and evaluation remain unchanged; supplemental driver/qualification receipts record the recovery. Reader-native prompt and generation tokens are retained. Cached original timing fields omit some preprocessing/IPC costs; measured end-to-end latency remains null. No serving-latency conclusion is justified from those fields.

## Continuation decisions

- D stays frozen with LLMLingua conditions explicitly pending. No existing prediction, score, source selection or status artifact was overwritten for this review.

- E compares C0 raw top-10-pool passages and C2 exact query-ranked top-eight sentences; C1 remains pending. The QASPER/Mistral SARA specialist C3 has its own adapted no-evidence baseline and is reported separately. Because E uses width 10 while D uses width 20, controlled compressor claims must come from within E.

- F uses the five manually reviewed actual C actions selected before D/E outcomes; the same settings are tested on every dataset with Qwen and BM25/MiniLM/SFR. Their selection remains exploratory. D does not trigger a silent reader/retriever change.

- G keeps its prescribed Qwen/SFR hierarchy controls. H/I are saved-result analyses only. Prospective prompt/parser changes, a larger confirmation cohort, or dependency-qualified token compression require a new output version.

## Complete 333-condition summary

The complete fixed reader × selector × granularity × budget table follows. Metric scales remain dataset-specific; there is no pooled QA/verification overall score.

| dataset   | reader   | evidence_granularity   | selector   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |   generated_tokens |   invalid_label_rate |
|:----------|:---------|:-----------------------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|-------------------:|---------------------:|
| HotpotQA  | Llama    | NONE                   | none       | 0        |     0.094824 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       75.440000 |          18.780000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | hybrid     | 1024     |     0.397767 |           0.302943 | 0.640000 | 0.060000 |               789.980000 |      865.420000 |          14.340000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | hybrid     | 256      |     0.285376 |           0.190552 | 0.560000 | 0.060000 |               149.220000 |      224.660000 |          15.040000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | hybrid     | 512      |     0.402096 |           0.307272 | 0.640000 | 0.040000 |               360.740000 |      436.180000 |          13.400000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | hybrid     | FULL     |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 |     1369.020000 |          13.880000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | lexical    | 1024     |     0.335851 |           0.241027 | 0.560000 | 0.100000 |               775.300000 |      850.740000 |          15.140000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | lexical    | 256      |     0.280136 |           0.185312 | 0.520000 | 0.060000 |               138.880000 |      214.320000 |          14.940000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | lexical    | 512      |     0.353926 |           0.259102 | 0.560000 | 0.120000 |               357.480000 |      432.920000 |          16.760000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | lexical    | FULL     |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 |     1369.020000 |          13.880000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | semantic   | 1024     |     0.424796 |           0.329972 | 0.700000 | 0.040000 |               762.960000 |      838.400000 |          13.920000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | semantic   | 256      |     0.284481 |           0.189657 | 0.580000 | 0.100000 |               157.560000 |      233.000000 |          15.920000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | semantic   | 512      |     0.386535 |           0.291711 | 0.660000 | 0.020000 |               370.380000 |      445.820000 |          13.860000 |           nan        |
| HotpotQA  | Llama    | PASSAGE                | semantic   | FULL     |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 |     1369.020000 |          13.880000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | hybrid     | 1024     |     0.340284 |           0.245460 | 0.580000 | 0.100000 |               829.680000 |      905.120000 |          16.220000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | hybrid     | 256      |     0.321213 |           0.226389 | 0.580000 | 0.120000 |               193.500000 |      268.940000 |          16.380000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | hybrid     | 512      |     0.378682 |           0.283858 | 0.680000 | 0.080000 |               404.120000 |      479.580000 |          14.280000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | hybrid     | FULL     |     0.443019 |           0.348195 | 0.680000 | 0.040000 |              1295.200000 |     1370.640000 |          13.820000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | lexical    | 1024     |     0.354771 |           0.259947 | 0.660000 | 0.080000 |               829.580000 |      905.020000 |          15.120000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | lexical    | 256      |     0.301734 |           0.206910 | 0.540000 | 0.100000 |               191.280000 |      266.760000 |          15.420000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | lexical    | 512      |     0.313591 |           0.218767 | 0.520000 | 0.120000 |               406.480000 |      481.920000 |          15.160000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | lexical    | FULL     |     0.443019 |           0.348195 | 0.680000 | 0.040000 |              1295.200000 |     1370.640000 |          13.820000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | semantic   | 1024     |     0.354486 |           0.259662 | 0.620000 | 0.100000 |               829.440000 |      904.880000 |          16.240000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | semantic   | 256      |     0.362092 |           0.267268 | 0.640000 | 0.080000 |               188.060000 |      263.500000 |          15.020000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | semantic   | 512      |     0.368876 |           0.274052 | 0.680000 | 0.060000 |               407.420000 |      482.860000 |          13.860000 |           nan        |
| HotpotQA  | Llama    | PROPOSITION            | semantic   | FULL     |     0.443019 |           0.348195 | 0.680000 | 0.040000 |              1295.200000 |     1370.640000 |          13.820000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | hybrid     | 1024     |     0.376278 |           0.281454 | 0.680000 | 0.060000 |               830.140000 |      905.580000 |          16.480000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | hybrid     | 256      |     0.344672 |           0.249848 | 0.600000 | 0.100000 |               195.180000 |      270.620000 |          16.120000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | hybrid     | 512      |     0.361677 |           0.266853 | 0.640000 | 0.080000 |               400.260000 |      475.720000 |          14.540000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | hybrid     | FULL     |     0.426510 |           0.331686 | 0.700000 | 0.040000 |              1293.560000 |     1369.000000 |          14.240000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | lexical    | 1024     |     0.346009 |           0.251185 | 0.680000 | 0.060000 |               833.860000 |      909.300000 |          16.520000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | lexical    | 256      |     0.316225 |           0.221401 | 0.540000 | 0.060000 |               191.780000 |      267.260000 |          15.040000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | lexical    | 512      |     0.319374 |           0.224550 | 0.520000 | 0.100000 |               407.000000 |      482.440000 |          14.760000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | lexical    | FULL     |     0.426510 |           0.331686 | 0.700000 | 0.040000 |              1293.560000 |     1369.000000 |          14.240000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | semantic   | 1024     |     0.335331 |           0.240507 | 0.620000 | 0.100000 |               830.500000 |      905.940000 |          15.640000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | semantic   | 256      |     0.400342 |           0.305518 | 0.660000 | 0.060000 |               190.340000 |      265.780000 |          14.260000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | semantic   | 512      |     0.366556 |           0.271732 | 0.700000 | 0.060000 |               403.920000 |      479.360000 |          14.200000 |           nan        |
| HotpotQA  | Llama    | SENTENCE               | semantic   | FULL     |     0.426510 |           0.331686 | 0.700000 | 0.040000 |              1293.560000 |     1369.000000 |          14.240000 |           nan        |
| HotpotQA  | Mistral  | NONE                   | none       | 0        |     0.100453 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       53.360000 |          28.420000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | hybrid     | 1024     |     0.317028 |           0.216575 | 0.520000 | 0.200000 |               933.340000 |      987.200000 |          25.320000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | hybrid     | 256      |     0.243841 |           0.143388 | 0.540000 | 0.220000 |               176.200000 |      230.160000 |          23.160000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | hybrid     | 512      |     0.303408 |           0.202955 | 0.580000 | 0.140000 |               426.680000 |      480.620000 |          25.860000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | hybrid     | FULL     |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 |     1581.120000 |          26.980000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | lexical    | 1024     |     0.275823 |           0.175370 | 0.460000 | 0.200000 |               916.400000 |      970.240000 |          24.500000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | lexical    | 256      |     0.228822 |           0.128369 | 0.540000 | 0.160000 |               163.660000 |      217.560000 |          25.440000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | lexical    | 512      |     0.228306 |           0.127853 | 0.480000 | 0.140000 |               422.820000 |      476.800000 |          28.760000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | lexical    | FULL     |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 |     1581.120000 |          26.980000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | semantic   | 1024     |     0.309067 |           0.208614 | 0.560000 | 0.200000 |               903.240000 |      957.060000 |          24.780000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | semantic   | 256      |     0.238869 |           0.138417 | 0.440000 | 0.240000 |               185.200000 |      239.020000 |          25.280000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | semantic   | 512      |     0.324281 |           0.223828 | 0.580000 | 0.140000 |               439.180000 |      493.060000 |          23.000000 |           nan        |
| HotpotQA  | Mistral  | PASSAGE                | semantic   | FULL     |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 |     1581.120000 |          26.980000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | hybrid     | 1024     |     0.285508 |           0.185055 | 0.560000 | 0.180000 |              1000.960000 |     1054.720000 |          29.060000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | hybrid     | 256      |     0.248974 |           0.148521 | 0.440000 | 0.200000 |               232.940000 |      286.480000 |          26.720000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | hybrid     | 512      |     0.276733 |           0.176280 | 0.560000 | 0.160000 |               487.840000 |      541.500000 |          28.040000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | hybrid     | FULL     |     0.320597 |           0.220144 | 0.580000 | 0.240000 |              1577.160000 |     1630.980000 |          27.420000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | lexical    | 1024     |     0.275703 |           0.175250 | 0.500000 | 0.200000 |              1001.160000 |     1054.920000 |          31.480000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | lexical    | 256      |     0.211240 |           0.110787 | 0.400000 | 0.140000 |               230.680000 |      284.240000 |          26.860000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | lexical    | 512      |     0.226394 |           0.125941 | 0.460000 | 0.200000 |               489.800000 |      543.360000 |          28.200000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | lexical    | FULL     |     0.320597 |           0.220144 | 0.580000 | 0.240000 |              1577.160000 |     1630.980000 |          27.420000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | semantic   | 1024     |     0.311296 |           0.210843 | 0.560000 | 0.180000 |              1004.240000 |     1058.060000 |          24.920000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | semantic   | 256      |     0.267987 |           0.167534 | 0.540000 | 0.200000 |               228.020000 |      281.660000 |          25.320000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | semantic   | 512      |     0.288681 |           0.188229 | 0.620000 | 0.120000 |               493.260000 |      546.940000 |          24.040000 |           nan        |
| HotpotQA  | Mistral  | PROPOSITION            | semantic   | FULL     |     0.320597 |           0.220144 | 0.580000 | 0.240000 |              1577.160000 |     1630.980000 |          27.420000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | hybrid     | 1024     |     0.281984 |           0.181531 | 0.560000 | 0.220000 |               997.840000 |     1051.620000 |          29.220000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | hybrid     | 256      |     0.244431 |           0.143978 | 0.500000 | 0.220000 |               234.300000 |      287.860000 |          26.760000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | hybrid     | 512      |     0.275711 |           0.175258 | 0.560000 | 0.180000 |               482.380000 |      536.040000 |          26.060000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | hybrid     | FULL     |     0.313041 |           0.212588 | 0.560000 | 0.240000 |              1568.640000 |     1622.460000 |          29.240000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | lexical    | 1024     |     0.297737 |           0.197284 | 0.540000 | 0.180000 |              1003.620000 |     1057.420000 |          29.220000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | lexical    | 256      |     0.218377 |           0.117924 | 0.420000 | 0.180000 |               230.080000 |      283.680000 |          28.000000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | lexical    | 512      |     0.234832 |           0.134379 | 0.500000 | 0.200000 |               489.400000 |      543.000000 |          28.680000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | lexical    | FULL     |     0.313041 |           0.212588 | 0.560000 | 0.240000 |              1568.640000 |     1622.460000 |          29.240000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | semantic   | 1024     |     0.288463 |           0.188010 | 0.540000 | 0.200000 |              1001.600000 |     1055.440000 |          28.720000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | semantic   | 256      |     0.251836 |           0.151383 | 0.600000 | 0.200000 |               229.500000 |      283.180000 |          28.680000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | semantic   | 512      |     0.298207 |           0.197754 | 0.600000 | 0.120000 |               487.260000 |      540.960000 |          25.580000 |           nan        |
| HotpotQA  | Mistral  | SENTENCE               | semantic   | FULL     |     0.313041 |           0.212588 | 0.560000 | 0.240000 |              1568.640000 |     1622.460000 |          29.240000 |           nan        |
| HotpotQA  | Qwen     | NONE                   | none       | 0        |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       52.800000 |          13.820000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | hybrid     | 1024     |     0.447304 |           0.297054 | 0.540000 | 0.060000 |               825.180000 |      877.980000 |          12.100000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | hybrid     | 256      |     0.384781 |           0.234531 | 0.520000 | 0.040000 |               157.200000 |      210.020000 |          13.440000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | hybrid     | 512      |     0.387291 |           0.237041 | 0.480000 | 0.060000 |               378.240000 |      431.060000 |          13.380000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | hybrid     | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | lexical    | 1024     |     0.378796 |           0.228546 | 0.460000 | 0.060000 |               809.560000 |      862.360000 |          12.800000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | lexical    | 256      |     0.367020 |           0.216770 | 0.480000 | 0.040000 |               145.900000 |      198.720000 |          14.040000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | lexical    | 512      |     0.324755 |           0.174505 | 0.420000 | 0.100000 |               374.220000 |      427.020000 |          15.340000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | lexical    | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | semantic   | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |          11.980000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | semantic   | 256      |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 |      218.700000 |          12.440000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | semantic   | 512      |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |      441.920000 |          12.140000 |           nan        |
| HotpotQA  | Qwen     | PASSAGE                | semantic   | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |          13.520000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | hybrid     | 1024     |     0.442534 |           0.292285 | 0.560000 | 0.060000 |               864.560000 |      917.360000 |          11.440000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | hybrid     | 256      |     0.357872 |           0.207622 | 0.440000 | 0.060000 |               203.020000 |      255.840000 |          11.140000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | hybrid     | 512      |     0.377521 |           0.227271 | 0.520000 | 0.040000 |               423.000000 |      475.820000 |          12.840000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | hybrid     | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |          12.640000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | lexical    | 1024     |     0.433383 |           0.283134 | 0.600000 | 0.020000 |               865.480000 |      918.280000 |          12.460000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | lexical    | 256      |     0.316006 |           0.165757 | 0.360000 | 0.040000 |               199.800000 |      252.660000 |          11.580000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | lexical    | 512      |     0.316340 |           0.166090 | 0.380000 | 0.080000 |               424.380000 |      477.180000 |          13.540000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | lexical    | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |          12.640000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | semantic   | 1024     |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |      917.840000 |          10.760000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | semantic   | 256      |     0.407943 |           0.257694 | 0.520000 | 0.080000 |               198.300000 |      251.100000 |          12.960000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | semantic   | 512      |     0.430585 |           0.280335 | 0.520000 | 0.060000 |               426.660000 |      479.460000 |          12.460000 |           nan        |
| HotpotQA  | Qwen     | PROPOSITION            | semantic   | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |          12.640000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | hybrid     | 1024     |     0.470987 |           0.320737 | 0.600000 | 0.040000 |               865.600000 |      918.400000 |          10.980000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | hybrid     | 256      |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 |      257.920000 |          11.260000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | hybrid     | 512      |     0.396624 |           0.246374 | 0.500000 | 0.060000 |               419.780000 |      472.600000 |          12.760000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | hybrid     | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |          13.180000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | lexical    | 1024     |     0.399654 |           0.249405 | 0.520000 | 0.060000 |               870.440000 |      923.240000 |          11.860000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | lexical    | 256      |     0.348744 |           0.198495 | 0.380000 | 0.020000 |               200.580000 |      253.420000 |          11.980000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | lexical    | 512      |     0.337007 |           0.186758 | 0.400000 | 0.060000 |               425.340000 |      478.140000 |          12.560000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | lexical    | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |          13.180000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | semantic   | 1024     |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |      919.540000 |          11.060000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | semantic   | 256      |     0.421265 |           0.271015 | 0.540000 | 0.060000 |               201.160000 |      253.960000 |          12.480000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | semantic   | 512      |     0.420340 |           0.270090 | 0.560000 | 0.060000 |               423.360000 |      476.160000 |          13.120000 |           nan        |
| HotpotQA  | Qwen     | SENTENCE               | semantic   | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |          13.180000 |           nan        |
| QASPER    | Llama    | NONE                   | none       | 0        |     0.071760 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       64.940000 |          17.720000 |           nan        |
| QASPER    | Llama    | PASSAGE                | hybrid     | 1024     |     0.194788 |           0.123028 | 0.600000 | 0.120000 |               757.380000 |      822.900000 |          31.480000 |           nan        |
| QASPER    | Llama    | PASSAGE                | hybrid     | 256      |     0.071760 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       64.940000 |          17.720000 |           nan        |
| QASPER    | Llama    | PASSAGE                | hybrid     | 512      |     0.151562 |           0.079802 | 0.480000 | 0.180000 |               263.780000 |      329.380000 |          18.820000 |           nan        |
| QASPER    | Llama    | PASSAGE                | hybrid     | FULL     |     0.278155 |           0.206395 | 0.680000 | 0.140000 |              4076.580000 |     4141.660000 |          42.800000 |           nan        |
| QASPER    | Llama    | PASSAGE                | lexical    | 1024     |     0.185994 |           0.114234 | 0.580000 | 0.120000 |               757.880000 |      823.480000 |          29.800000 |           nan        |
| QASPER    | Llama    | PASSAGE                | lexical    | 256      |     0.066760 |          -0.005000 | 0.000000 | 0.020000 |                 3.320000 |       68.260000 |          17.960000 |           nan        |
| QASPER    | Llama    | PASSAGE                | lexical    | 512      |     0.145517 |           0.073757 | 0.440000 | 0.160000 |               258.680000 |      324.360000 |          18.840000 |           nan        |
| QASPER    | Llama    | PASSAGE                | lexical    | FULL     |     0.278155 |           0.206395 | 0.680000 | 0.140000 |              4076.580000 |     4141.660000 |          42.800000 |           nan        |
| QASPER    | Llama    | PASSAGE                | semantic   | 1024     |     0.198886 |           0.127126 | 0.580000 | 0.180000 |               759.700000 |      825.280000 |          30.800000 |           nan        |
| QASPER    | Llama    | PASSAGE                | semantic   | 256      |     0.085587 |           0.013827 | 0.040000 | 0.020000 |                18.560000 |       83.540000 |          17.900000 |           nan        |
| QASPER    | Llama    | PASSAGE                | semantic   | 512      |     0.131191 |           0.059432 | 0.440000 | 0.200000 |               262.660000 |      328.320000 |          21.800000 |           nan        |
| QASPER    | Llama    | PASSAGE                | semantic   | FULL     |     0.278155 |           0.206395 | 0.680000 | 0.140000 |              4076.580000 |     4141.660000 |          42.800000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | hybrid     | 1024     |     0.238638 |           0.166879 | 0.700000 | 0.120000 |               857.120000 |      922.120000 |          26.980000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | hybrid     | 256      |     0.193223 |           0.121463 | 0.560000 | 0.140000 |               200.720000 |      265.720000 |          19.020000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | hybrid     | 512      |     0.197401 |           0.125641 | 0.580000 | 0.120000 |               419.640000 |      484.640000 |          21.660000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | hybrid     | FULL     |     0.287408 |           0.215648 | 0.700000 | 0.120000 |              4078.920000 |     4144.000000 |          38.880000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | lexical    | 1024     |     0.224114 |           0.152355 | 0.680000 | 0.120000 |               851.480000 |      916.500000 |          25.400000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | lexical    | 256      |     0.122986 |           0.051226 | 0.420000 | 0.180000 |               201.680000 |      266.700000 |          17.760000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | lexical    | 512      |     0.198696 |           0.126936 | 0.500000 | 0.180000 |               416.360000 |      481.380000 |          21.640000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | lexical    | FULL     |     0.287408 |           0.215648 | 0.700000 | 0.120000 |              4078.920000 |     4144.000000 |          38.880000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | semantic   | 1024     |     0.206456 |           0.134696 | 0.620000 | 0.120000 |               856.400000 |      921.420000 |          29.360000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | semantic   | 256      |     0.165494 |           0.093734 | 0.500000 | 0.160000 |               200.700000 |      265.720000 |          20.480000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | semantic   | 512      |     0.193099 |           0.121339 | 0.560000 | 0.120000 |               419.920000 |      484.960000 |          22.540000 |           nan        |
| QASPER    | Llama    | PROPOSITION            | semantic   | FULL     |     0.287408 |           0.215648 | 0.700000 | 0.120000 |              4078.920000 |     4144.000000 |          38.880000 |           nan        |
| QASPER    | Llama    | SENTENCE               | hybrid     | 1024     |     0.245748 |           0.173989 | 0.700000 | 0.120000 |               859.880000 |      924.860000 |          27.280000 |           nan        |
| QASPER    | Llama    | SENTENCE               | hybrid     | 256      |     0.192667 |           0.120907 | 0.540000 | 0.120000 |               198.580000 |      263.580000 |          18.220000 |           nan        |
| QASPER    | Llama    | SENTENCE               | hybrid     | 512      |     0.207772 |           0.136012 | 0.580000 | 0.120000 |               419.880000 |      484.900000 |          23.000000 |           nan        |
| QASPER    | Llama    | SENTENCE               | hybrid     | FULL     |     0.289146 |           0.217386 | 0.720000 | 0.120000 |              4079.320000 |     4144.400000 |          39.440000 |           nan        |
| QASPER    | Llama    | SENTENCE               | lexical    | 1024     |     0.246513 |           0.174753 | 0.700000 | 0.120000 |               854.540000 |      919.560000 |          27.240000 |           nan        |
| QASPER    | Llama    | SENTENCE               | lexical    | 256      |     0.123232 |           0.051472 | 0.420000 | 0.180000 |               203.060000 |      268.100000 |          17.760000 |           nan        |
| QASPER    | Llama    | SENTENCE               | lexical    | 512      |     0.193243 |           0.121484 | 0.480000 | 0.160000 |               419.380000 |      484.400000 |          21.340000 |           nan        |
| QASPER    | Llama    | SENTENCE               | lexical    | FULL     |     0.289146 |           0.217386 | 0.720000 | 0.120000 |              4079.320000 |     4144.400000 |          39.440000 |           nan        |
| QASPER    | Llama    | SENTENCE               | semantic   | 1024     |     0.209731 |           0.137972 | 0.620000 | 0.120000 |               858.520000 |      923.540000 |          28.120000 |           nan        |
| QASPER    | Llama    | SENTENCE               | semantic   | 256      |     0.179536 |           0.107776 | 0.540000 | 0.140000 |               198.740000 |      263.740000 |          20.720000 |           nan        |
| QASPER    | Llama    | SENTENCE               | semantic   | 512      |     0.196246 |           0.124487 | 0.580000 | 0.120000 |               420.260000 |      485.300000 |          23.220000 |           nan        |
| QASPER    | Llama    | SENTENCE               | semantic   | FULL     |     0.289146 |           0.217386 | 0.720000 | 0.120000 |              4079.320000 |     4144.400000 |          39.440000 |           nan        |
| QASPER    | Mistral  | NONE                   | none       | 0        |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.660000 |          22.040000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | hybrid     | 1024     |     0.198421 |           0.138616 | 0.700000 | 0.120000 |               845.940000 |      887.740000 |          42.400000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | hybrid     | 256      |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.660000 |          22.040000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | hybrid     | 512      |     0.141074 |           0.081269 | 0.500000 | 0.160000 |               294.120000 |      335.800000 |          28.800000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | hybrid     | FULL     |     0.180881 |           0.121076 | 0.680000 | 0.100000 |              4553.040000 |     4594.840000 |          64.840000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | lexical    | 1024     |     0.192404 |           0.132599 | 0.680000 | 0.100000 |               846.940000 |      888.720000 |          41.240000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | lexical    | 256      |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 4.000000 |       45.640000 |          22.180000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | lexical    | 512      |     0.132969 |           0.073163 | 0.480000 | 0.180000 |               289.080000 |      330.880000 |          28.820000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | lexical    | FULL     |     0.180881 |           0.121076 | 0.680000 | 0.100000 |              4553.040000 |     4594.840000 |          64.840000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | semantic   | 1024     |     0.192379 |           0.132573 | 0.700000 | 0.100000 |               848.440000 |      890.280000 |          41.020000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | semantic   | 256      |     0.064574 |           0.004769 | 0.060000 | 0.020000 |                19.660000 |       61.320000 |          22.500000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | semantic   | 512      |     0.140820 |           0.081014 | 0.540000 | 0.180000 |               289.480000 |      331.260000 |          33.440000 |           nan        |
| QASPER    | Mistral  | PASSAGE                | semantic   | FULL     |     0.180881 |           0.121076 | 0.680000 | 0.100000 |              4553.040000 |     4594.840000 |          64.840000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | hybrid     | 1024     |     0.166722 |           0.106917 | 0.660000 | 0.140000 |              1003.940000 |     1045.680000 |          46.880000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | hybrid     | 256      |     0.146353 |           0.086548 | 0.520000 | 0.180000 |               234.820000 |      276.660000 |          28.000000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | hybrid     | 512      |     0.171341 |           0.111536 | 0.640000 | 0.140000 |               491.800000 |      533.500000 |          33.140000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | hybrid     | FULL     |     0.183302 |           0.123496 | 0.640000 | 0.140000 |              4820.680000 |     4862.480000 |          60.800000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | lexical    | 1024     |     0.191482 |           0.131676 | 0.660000 | 0.120000 |              1000.640000 |     1042.400000 |          43.680000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | lexical    | 256      |     0.132724 |           0.072919 | 0.460000 | 0.160000 |               236.540000 |      278.240000 |          28.200000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | lexical    | 512      |     0.192500 |           0.132695 | 0.580000 | 0.140000 |               489.100000 |      530.840000 |          28.980000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | lexical    | FULL     |     0.183302 |           0.123496 | 0.640000 | 0.140000 |              4820.680000 |     4862.480000 |          60.800000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | semantic   | 1024     |     0.187808 |           0.128003 | 0.660000 | 0.120000 |              1001.100000 |     1042.880000 |          41.880000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | semantic   | 256      |     0.170798 |           0.110992 | 0.560000 | 0.180000 |               234.600000 |      276.460000 |          30.120000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | semantic   | 512      |     0.176373 |           0.116567 | 0.600000 | 0.140000 |               491.960000 |      533.800000 |          32.880000 |           nan        |
| QASPER    | Mistral  | PROPOSITION            | semantic   | FULL     |     0.183302 |           0.123496 | 0.640000 | 0.140000 |              4820.680000 |     4862.480000 |          60.800000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | hybrid     | 1024     |     0.170431 |           0.110626 | 0.620000 | 0.160000 |              1005.480000 |     1047.220000 |          43.840000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | hybrid     | 256      |     0.151635 |           0.091829 | 0.520000 | 0.180000 |               231.960000 |      273.760000 |          26.680000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | hybrid     | 512      |     0.174255 |           0.114449 | 0.620000 | 0.160000 |               491.180000 |      532.860000 |          32.980000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | hybrid     | FULL     |     0.194480 |           0.134675 | 0.660000 | 0.160000 |              4812.160000 |     4853.960000 |          63.520000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | lexical    | 1024     |     0.200750 |           0.140944 | 0.640000 | 0.140000 |              1002.280000 |     1044.020000 |          40.740000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | lexical    | 256      |     0.131516 |           0.071710 | 0.460000 | 0.160000 |               237.260000 |      278.980000 |          28.140000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | lexical    | 512      |     0.189919 |           0.130114 | 0.540000 | 0.160000 |               491.680000 |      533.420000 |          30.460000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | lexical    | FULL     |     0.194480 |           0.134675 | 0.660000 | 0.160000 |              4812.160000 |     4853.960000 |          63.520000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | semantic   | 1024     |     0.181724 |           0.121918 | 0.660000 | 0.100000 |              1002.420000 |     1044.200000 |          41.780000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | semantic   | 256      |     0.179511 |           0.119706 | 0.600000 | 0.160000 |               232.080000 |      273.920000 |          30.400000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | semantic   | 512      |     0.186299 |           0.126494 | 0.620000 | 0.140000 |               491.840000 |      533.660000 |          32.260000 |           nan        |
| QASPER    | Mistral  | SENTENCE               | semantic   | FULL     |     0.194480 |           0.134675 | 0.660000 | 0.160000 |              4812.160000 |     4853.960000 |          63.520000 |           nan        |
| QASPER    | Qwen     | NONE                   | none       | 0        |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |          18.160000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | hybrid     | 1024     |     0.212562 |           0.121334 | 0.640000 | 0.060000 |               769.600000 |      812.160000 |          21.060000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | hybrid     | 256      |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |          18.160000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | hybrid     | 512      |     0.171876 |           0.080648 | 0.460000 | 0.120000 |               267.660000 |      310.300000 |          17.720000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | hybrid     | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | lexical    | 1024     |     0.221855 |           0.130627 | 0.640000 | 0.060000 |               769.600000 |      812.240000 |          20.320000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | lexical    | 256      |     0.086783 |          -0.004444 | 0.000000 | 0.020000 |                 3.400000 |       45.380000 |          18.080000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | lexical    | 512      |     0.158010 |           0.066782 | 0.440000 | 0.180000 |               262.400000 |      305.120000 |          18.080000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | lexical    | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | semantic   | 1024     |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |      816.520000 |          20.380000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | semantic   | 256      |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 |       60.740000 |          17.960000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | semantic   | 512      |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 |      309.800000 |          19.320000 |           nan        |
| QASPER    | Qwen     | PASSAGE                | semantic   | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |          21.200000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | hybrid     | 1024     |     0.202180 |           0.110952 | 0.600000 | 0.200000 |               870.100000 |      912.140000 |          19.560000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | hybrid     | 256      |     0.195327 |           0.104099 | 0.480000 | 0.220000 |               202.760000 |      244.800000 |          18.140000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | hybrid     | 512      |     0.199701 |           0.108474 | 0.520000 | 0.200000 |               424.420000 |      466.460000 |          19.160000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | hybrid     | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |          21.460000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | lexical    | 1024     |     0.204258 |           0.113031 | 0.560000 | 0.140000 |               863.120000 |      905.180000 |          19.020000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | lexical    | 256      |     0.149739 |           0.058511 | 0.400000 | 0.220000 |               203.380000 |      245.440000 |          16.600000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | lexical    | 512      |     0.168587 |           0.077359 | 0.500000 | 0.180000 |               421.040000 |      463.100000 |          17.860000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | lexical    | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |          21.460000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | semantic   | 1024     |     0.200596 |           0.109368 | 0.560000 | 0.140000 |               870.780000 |      912.860000 |          20.400000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | semantic   | 256      |     0.209378 |           0.118151 | 0.520000 | 0.160000 |               203.400000 |      245.460000 |          19.300000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | semantic   | 512      |     0.189278 |           0.098050 | 0.520000 | 0.180000 |               426.840000 |      468.920000 |          21.300000 |           nan        |
| QASPER    | Qwen     | PROPOSITION            | semantic   | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |          21.460000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | hybrid     | 1024     |     0.200471 |           0.109243 | 0.580000 | 0.180000 |               872.980000 |      915.000000 |          19.560000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | hybrid     | 256      |     0.197049 |           0.105822 | 0.500000 | 0.200000 |               200.560000 |      242.600000 |          17.620000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | hybrid     | 512      |     0.196996 |           0.105768 | 0.540000 | 0.200000 |               424.560000 |      466.620000 |          18.960000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | hybrid     | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |          21.160000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | lexical    | 1024     |     0.195360 |           0.104133 | 0.520000 | 0.140000 |               866.180000 |      908.240000 |          19.500000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | lexical    | 256      |     0.152126 |           0.060898 | 0.420000 | 0.200000 |               204.760000 |      246.840000 |          17.040000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | lexical    | 512      |     0.192098 |           0.100871 | 0.520000 | 0.180000 |               424.040000 |      466.100000 |          17.480000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | lexical    | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |          21.160000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | semantic   | 1024     |     0.204816 |           0.113589 | 0.560000 | 0.140000 |               872.960000 |      915.020000 |          20.680000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | semantic   | 256      |     0.211284 |           0.120056 | 0.560000 | 0.160000 |               201.460000 |      243.500000 |          19.660000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | semantic   | 512      |     0.191808 |           0.100580 | 0.560000 | 0.180000 |               427.160000 |      469.240000 |          21.300000 |           nan        |
| QASPER    | Qwen     | SENTENCE               | semantic   | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |          21.160000 |           nan        |
| SciFact   | Llama    | NONE                   | none       | 0        |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       81.220000 |         110.440000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               618.480000 |      699.700000 |          94.060000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                19.500000 |      100.720000 |         107.060000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               237.400000 |      318.640000 |          84.040000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | hybrid     | FULL     |     0.040000 |           0.040000 | 0.040000 | 0.000000 |              4819.100000 |     4900.320000 |         109.280000 |             0.920000 |
| SciFact   | Llama    | PASSAGE                | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               637.940000 |      719.160000 |          94.500000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                13.340000 |       94.560000 |         108.040000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               231.540000 |      312.780000 |          84.060000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | lexical    | FULL     |     0.040000 |           0.040000 | 0.040000 | 0.000000 |              4819.100000 |     4900.320000 |         109.280000 |             0.920000 |
| SciFact   | Llama    | PASSAGE                | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               653.920000 |      735.160000 |          99.740000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                19.140000 |      100.360000 |         106.500000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               234.680000 |      315.900000 |          86.740000 |             1.000000 |
| SciFact   | Llama    | PASSAGE                | semantic   | FULL     |     0.040000 |           0.040000 | 0.040000 | 0.000000 |              4819.100000 |     4900.320000 |         109.280000 |             0.920000 |
| SciFact   | Llama    | PROPOSITION            | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               799.940000 |      881.220000 |         100.300000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               183.600000 |      264.820000 |          87.980000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               389.440000 |      470.680000 |          91.860000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | hybrid     | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4810.620000 |     4891.840000 |         109.060000 |             0.960000 |
| SciFact   | Llama    | PROPOSITION            | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               796.900000 |      878.180000 |         105.720000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               185.380000 |      266.720000 |          92.960000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               389.320000 |      470.560000 |          88.640000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | lexical    | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4810.620000 |     4891.840000 |         109.060000 |             0.960000 |
| SciFact   | Llama    | PROPOSITION            | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               801.880000 |      883.100000 |         108.900000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               185.500000 |      266.720000 |          89.600000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               385.040000 |      466.300000 |          92.980000 |             1.000000 |
| SciFact   | Llama    | PROPOSITION            | semantic   | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4810.620000 |     4891.840000 |         109.060000 |             0.960000 |
| SciFact   | Llama    | SENTENCE               | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               795.580000 |      876.860000 |         102.880000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               185.400000 |      266.640000 |          84.480000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               390.240000 |      471.480000 |          89.860000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | hybrid     | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4826.640000 |     4907.860000 |         107.980000 |             0.960000 |
| SciFact   | Llama    | SENTENCE               | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               798.500000 |      879.780000 |         106.200000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               185.720000 |      266.980000 |          92.260000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               386.540000 |      467.780000 |          94.480000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | lexical    | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4826.640000 |     4907.860000 |         107.980000 |             0.960000 |
| SciFact   | Llama    | SENTENCE               | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               805.480000 |      886.700000 |         108.660000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               184.520000 |      265.740000 |          87.780000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               386.780000 |      468.040000 |          92.180000 |             1.000000 |
| SciFact   | Llama    | SENTENCE               | semantic   | FULL     |     0.020000 |           0.020000 | 0.020000 | 0.000000 |              4826.640000 |     4907.860000 |         107.980000 |             0.960000 |
| SciFact   | Mistral  | NONE                   | none       | 0        |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       66.780000 |          79.860000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               756.480000 |      823.340000 |          67.060000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                23.700000 |       90.480000 |          75.220000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               289.020000 |      355.860000 |          65.120000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | hybrid     | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              5917.040000 |     5984.000000 |          94.240000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               778.640000 |      845.520000 |          67.200000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                16.440000 |       83.220000 |          76.120000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               280.180000 |      346.960000 |          62.460000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | lexical    | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              5917.040000 |     5984.000000 |          94.240000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               800.320000 |      867.260000 |          70.680000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                23.440000 |       90.200000 |          78.320000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               285.200000 |      351.980000 |          70.500000 |             1.000000 |
| SciFact   | Mistral  | PASSAGE                | semantic   | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              5917.040000 |     5984.000000 |          94.240000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               998.120000 |     1065.160000 |          76.400000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               228.600000 |      295.500000 |          63.560000 |             0.980000 |
| SciFact   | Mistral  | PROPOSITION            | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               484.580000 |      551.540000 |          73.440000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | hybrid     | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6040.740000 |     6107.700000 |          92.780000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               995.920000 |     1062.860000 |          72.360000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               231.020000 |      298.040000 |          64.640000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               485.820000 |      552.840000 |          70.160000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | lexical    | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6040.740000 |     6107.700000 |          92.780000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               997.500000 |     1064.580000 |          80.180000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               228.980000 |      295.780000 |          65.000000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               476.440000 |      543.400000 |          71.820000 |             1.000000 |
| SciFact   | Mistral  | PROPOSITION            | semantic   | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6040.740000 |     6107.700000 |          92.780000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | hybrid     | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               991.240000 |     1058.300000 |          72.440000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | hybrid     | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               230.440000 |      297.340000 |          65.700000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | hybrid     | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               485.460000 |      552.420000 |          71.560000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | hybrid     | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6043.260000 |     6110.220000 |          94.680000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | lexical    | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               996.720000 |     1063.680000 |          72.760000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | lexical    | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               231.020000 |      298.040000 |          66.720000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | lexical    | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               481.980000 |      548.960000 |          69.020000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | lexical    | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6043.260000 |     6110.220000 |          94.680000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | semantic   | 1024     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              1000.180000 |     1067.300000 |          76.540000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | semantic   | 256      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               227.120000 |      293.900000 |          69.640000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | semantic   | 512      |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               478.300000 |      545.240000 |          72.640000 |             1.000000 |
| SciFact   | Mistral  | SENTENCE               | semantic   | FULL     |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              6043.260000 |     6110.220000 |          94.680000 |             1.000000 |
| SciFact   | Qwen     | NONE                   | none       | 0        |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       58.300000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | hybrid     | 1024     |     0.780000 |           0.420000 | 0.500000 | 0.080000 |               630.720000 |      689.020000 |           3.800000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | hybrid     | 256      |     0.360000 |           0.000000 | 0.020000 | 0.020000 |                19.620000 |       77.920000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | hybrid     | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               240.760000 |      299.080000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | hybrid     | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |           3.840000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | lexical    | 1024     |     0.740000 |           0.380000 | 0.520000 | 0.140000 |               652.960000 |      711.260000 |           3.960000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | lexical    | 256      |     0.340000 |          -0.020000 | 0.000000 | 0.020000 |                13.440000 |       71.740000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | lexical    | 512      |     0.620000 |           0.260000 | 0.360000 | 0.100000 |               234.140000 |      292.460000 |           3.920000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | lexical    | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |           3.840000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | semantic   | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |           3.680000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | semantic   | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 |       77.660000 |           3.800000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | semantic   | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 |      296.000000 |           3.840000 |             0.000000 |
| SciFact   | Qwen     | PASSAGE                | semantic   | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |           3.840000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | hybrid     | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               813.880000 |      872.240000 |           3.600000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | hybrid     | 256      |     0.700000 |           0.340000 | 0.480000 | 0.140000 |               186.620000 |      244.920000 |           3.640000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | hybrid     | 512      |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               396.000000 |      454.320000 |           3.640000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | hybrid     | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | lexical    | 1024     |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               810.360000 |      868.720000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | lexical    | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               188.080000 |      246.500000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | lexical    | 512      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               395.400000 |      453.720000 |           3.640000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | lexical    | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               819.320000 |      877.620000 |           3.480000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | semantic   | 256      |     0.660000 |           0.300000 | 0.420000 | 0.120000 |               188.540000 |      246.840000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | semantic   | 512      |     0.620000 |           0.260000 | 0.400000 | 0.140000 |               392.560000 |      450.900000 |           3.560000 |             0.000000 |
| SciFact   | Qwen     | PROPOSITION            | semantic   | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |           3.720000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | hybrid     | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               810.660000 |      869.020000 |           3.600000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | hybrid     | 256      |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 |      246.780000 |           3.680000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | hybrid     | 512      |     0.720000 |           0.360000 | 0.460000 | 0.100000 |               397.100000 |      455.420000 |           3.560000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | hybrid     | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | lexical    | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               812.440000 |      870.800000 |           3.680000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | lexical    | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               188.360000 |      246.700000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | lexical    | 512      |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               392.820000 |      451.140000 |           3.600000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | lexical    | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |           3.760000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               824.500000 |      882.800000 |           3.480000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | semantic   | 256      |     0.680000 |           0.320000 | 0.420000 | 0.100000 |               187.620000 |      245.920000 |           3.680000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | semantic   | 512      |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               394.620000 |      452.960000 |           3.680000 |             0.000000 |
| SciFact   | Qwen     | SENTENCE               | semantic   | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |           3.760000 |             0.000000 |

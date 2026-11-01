# Week 11 F — audited pilot results

2,400 available logical rows completed; 50 queries per dataset. Pending components: []. This is the frozen development pilot, not full-dataset confirmation.

Source/config/code identities, checksums, full action membership, budgets and same-reader/checkpoint HELP/HURT pairs passed. No scientific outputs were replaced.

## HotpotQA

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   |   budget | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|---------:|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   |     1024 | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   |     1024 | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   |     1024 | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 39d891f9fed0db18d8c2 | same-reader no evidence | 0.330022 | 0.230832 |  0.431392 |

## QASPER

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.249211 |           0.157984 | 0.720000 | 0.060000 |              4144.320000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.241667 |           0.150440 | 0.720000 | 0.060000 |              4135.880000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 0dc37be7e1b23dacb4e6 | same-reader no evidence | 0.157984 | 0.112419 |  0.203568 |

## SciFact

Metric: classification accuracy. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   |   budget | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|---------:|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   |     1024 | C0           |     0.760000 |           0.400000 | 0.520000 | 0.120000 |               655.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     |      256 | C0           |     0.740000 |           0.380000 | 0.460000 | 0.080000 |               188.280000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     |      256 | C0           |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 8603dec831a84395948a | same-reader no evidence | 0.400000 | 0.200000 |  0.580000 |

## Statistical interpretation

Use paired 10,000-resample bootstrap intervals (seed 42): QASPER paper clusters, other datasets query/claim clusters. Saved comparisons include same-reader no-evidence and matched raw-passage controls; D also includes matched Mistral reader controls. A positive point estimate alone is insufficient for superiority. Intervals are nominal and exploratory; selecting best conditions on this cohort adds selection bias. No mixed raw QA/verification aggregate is valid.

Paired result files: retriever_pairwise_df.csv.

## Flags

- empty_evidence_strict_prefix: 0 logical rows.
- invalid_verification_label: 0 logical rows.
- generation_cap_reached: 0 logical rows.
- unmapped_gold_annotation: 112 logical rows.

Empty evidence follows the frozen strict-prefix budget rule; an oversized first unit is not silently skipped. Generation-cap hits require inspection, not deletion. Invalid SciFact outputs retain the predefined strict scoring. Gold-mapping gaps affect evidence-recall interpretation. Exact source support does not imply answer correctness.

Cost: 519 physical calls for 2,400 logical rows. Cache hits retain conceptual per-policy LM costs. Cached original timing fields are not measured end-to-end latency; that field remains null.

Oracles are retrospective ceilings and do not establish a deployable controller. Specialist/public-reader surfaces are separated in the saved reports.

## Complete condition summary

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens | dataset   |   prompt_tokens |   generated_tokens |   invalid_label_rate |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|:----------|----------------:|-------------------:|---------------------:|
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 | HotpotQA  |      257.920000 |          11.260000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 | HotpotQA  |     1403.860000 |          13.520000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | HotpotQA  |       52.800000 |          13.820000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 | HotpotQA  |      257.920000 |          11.260000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 | HotpotQA  |      917.840000 |          10.760000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 | HotpotQA  |      851.160000 |          11.980000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 | HotpotQA  |     1403.860000 |          13.520000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 | HotpotQA  |      917.840000 |          10.760000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 | HotpotQA  |      919.540000 |          11.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 | HotpotQA  |      851.160000 |          11.980000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 | HotpotQA  |      851.160000 |          11.980000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 | HotpotQA  |      919.540000 |          11.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 | HotpotQA  |      919.540000 |          11.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 | HotpotQA  |      917.840000 |          10.760000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 | HotpotQA  |     1403.860000 |          13.520000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 | HotpotQA  |      257.920000 |          11.260000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.192792 |           0.101564 | 0.500000 | 0.220000 |               200.020000 | QASPER    |      242.060000 |          17.660000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.249211 |           0.157984 | 0.720000 | 0.060000 |              4144.320000 | QASPER    |     4186.440000 |          22.360000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       41.980000 |          18.160000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.192379 |           0.101151 | 0.480000 | 0.220000 |               200.360000 | QASPER    |      242.400000 |          17.380000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.203683 |           0.112456 | 0.600000 | 0.140000 |               872.640000 | QASPER    |      914.720000 |          20.400000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 | QASPER    |      816.520000 |          20.380000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.241667 |           0.150440 | 0.720000 | 0.060000 |              4135.880000 | QASPER    |     4178.020000 |          20.860000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.200596 |           0.109368 | 0.560000 | 0.140000 |               870.780000 | QASPER    |      912.860000 |          20.400000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.202240 |           0.111012 | 0.560000 | 0.160000 |               871.420000 | QASPER    |      913.480000 |          20.580000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.197917 |           0.106689 | 0.600000 | 0.100000 |               772.960000 | QASPER    |      815.600000 |          20.260000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.201220 |           0.109992 | 0.600000 | 0.100000 |               771.700000 | QASPER    |      814.320000 |          20.880000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.204816 |           0.113589 | 0.560000 | 0.140000 |               872.960000 | QASPER    |      915.020000 |          20.680000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.204941 |           0.113714 | 0.580000 | 0.140000 |               870.920000 | QASPER    |      912.980000 |          20.720000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.200371 |           0.109144 | 0.580000 | 0.160000 |               872.920000 | QASPER    |      914.980000 |          20.600000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 | QASPER    |     4186.440000 |          21.200000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.197049 |           0.105822 | 0.500000 | 0.200000 |               200.560000 | QASPER    |      242.600000 |          17.620000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.740000 |           0.380000 | 0.460000 | 0.080000 |               188.280000 | SciFact   |      246.580000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.640000 |           0.280000 | 0.360000 | 0.080000 |              4868.960000 | SciFact   |     4927.260000 |           3.880000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | SciFact   |       58.300000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.640000 |           0.280000 | 0.460000 | 0.180000 |               186.680000 | SciFact   |      244.980000 |           3.680000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.620000 |           0.260000 | 0.400000 | 0.140000 |               816.700000 | SciFact   |      875.000000 |           3.480000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 | SciFact   |      725.940000 |           3.680000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.700000 |           0.340000 | 0.460000 | 0.120000 |              4954.120000 | SciFact   |     5012.460000 |           3.880000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               819.320000 | SciFact   |      877.620000 |           3.480000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.660000 |           0.300000 | 0.460000 | 0.160000 |               815.220000 | SciFact   |      873.540000 |           3.600000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               657.200000 | SciFact   |      715.520000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.760000 |           0.400000 | 0.520000 | 0.120000 |               655.000000 | SciFact   |      713.320000 |           3.840000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               824.500000 | SciFact   |      882.800000 |           3.480000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 1024     | C0           |     0.640000 |           0.280000 | 0.400000 | 0.120000 |               817.740000 | SciFact   |      876.040000 |           3.520000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PROPOSITION            | semantic   | 1024     | C0           |     0.660000 |           0.300000 | 0.460000 | 0.160000 |               808.320000 | SciFact   |      866.640000 |           3.600000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 | SciFact   |     4994.740000 |           3.840000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | hybrid     | 256      | C0           |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 | SciFact   |      246.780000 |           3.680000 |             0.000000 |

## Post-hoc label-format diagnostic — not primary rescoring

SciFact strict scores require the whole stripped output to be one exact label. A label followed by an explanation is invalid under the frozen rule. Substantial format failures confound reader comparisons: low strict accuracy cannot be attributed solely to evidence utilization or reasoning. The separate anchored leading-label diagnostic accepts only an unambiguous label at the start, followed by a boundary/punctuation; it does not search explanations for labels. It was introduced after inspecting Mistral format failures and is exploratory. Primary scores, predictions, bootstrap comparisons and oracles remain unchanged. A prospective output-format protocol requires a new version.

| dataset   | reader   | action_id            | evidence_granularity   | selector   | budget   | compressor   |   n_queries |   strict_accuracy |   strict_format_rate |   leading_label_rate |   leading_label_diagnostic_accuracy |   cap_hit_rate |
|:----------|:---------|:---------------------|:-----------------------|:-----------|:---------|:-------------|------------:|------------------:|---------------------:|---------------------:|------------------------------------:|---------------:|
| SciFact   | Qwen     | 0ae4cd8d7bbf0c1f335b | SENTENCE               | hybrid     | 256      | C0           |          50 |          0.740000 |             1.000000 |             1.000000 |                            0.740000 |       0.000000 |
| SciFact   | Qwen     | 0dc37be7e1b23dacb4e6 | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.640000 |             1.000000 |             1.000000 |                            0.640000 |       0.000000 |
| SciFact   | Qwen     | 0e8bebd7162b7ff88ee7 | NONE                   | none       | 0        | C0           |          50 |          0.360000 |             1.000000 |             1.000000 |                            0.360000 |       0.000000 |
| SciFact   | Qwen     | 1e0768de16b50846bbb7 | SENTENCE               | hybrid     | 256      | C0           |          50 |          0.640000 |             1.000000 |             1.000000 |                            0.640000 |       0.000000 |
| SciFact   | Qwen     | 21945e4fa2f8705919f5 | PROPOSITION            | semantic   | 1024     | C0           |          50 |          0.620000 |             1.000000 |             1.000000 |                            0.620000 |       0.000000 |
| SciFact   | Qwen     | 39d891f9fed0db18d8c2 | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | 40719a9b7fa6061367c9 | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | 5ef43643d916e4c17721 | PROPOSITION            | semantic   | 1024     | C0           |          50 |          0.620000 |             1.000000 |             1.000000 |                            0.620000 |       0.000000 |
| SciFact   | Qwen     | 7aa17908c2f61a860df7 | SENTENCE               | semantic   | 1024     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | 82393b5263355fa073d6 | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | 8603dec831a84395948a | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.760000 |             1.000000 |             1.000000 |                            0.760000 |       0.000000 |
| SciFact   | Qwen     | a0788207676a3dd94c29 | SENTENCE               | semantic   | 1024     | C0           |          50 |          0.620000 |             1.000000 |             1.000000 |                            0.620000 |       0.000000 |
| SciFact   | Qwen     | c050bf1f46f443179fae | SENTENCE               | semantic   | 1024     | C0           |          50 |          0.640000 |             1.000000 |             1.000000 |                            0.640000 |       0.000000 |
| SciFact   | Qwen     | c3720d3c14d50b74107b | PROPOSITION            | semantic   | 1024     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | ecbe1962e8a80f2318cc | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | fc9352ab57c0ab40eb3b | SENTENCE               | hybrid     | 256      | C0           |          50 |          0.720000 |             1.000000 |             1.000000 |                            0.720000 |       0.000000 |

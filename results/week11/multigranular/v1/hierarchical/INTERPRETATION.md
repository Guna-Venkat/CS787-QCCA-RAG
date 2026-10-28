# Week 11 G — audited pilot results

3,150 available logical rows completed; 50 queries per dataset. Pending components: ['LLMLingua-2']. This is the frozen development pilot, not full-dataset confirmation.

Source/config/code identities, checksums, full action membership, budgets and same-reader/checkpoint HELP/HURT pairs passed. No scientific outputs were replaced.

## HotpotQA

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   |   budget | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|---------:|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   |     1024 | C0           |     0.487169 |           0.336919 | 0.640000 | 0.040000 |               864.260000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   |     1024 | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   |     1024 | C0           |     0.479379 |           0.329129 | 0.620000 | 0.040000 |               867.640000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 0962fbdcf76305eedb48 | same-reader no evidence | 0.336919 | 0.233908 |  0.441963 |

## QASPER

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.245822 |           0.154594 | 0.720000 | 0.080000 |              4146.420000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.243726 |           0.152498 | 0.720000 | 0.060000 |              4146.780000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 217a31e1b7342ff3de03 | same-reader no evidence | 0.154594 | 0.108768 |  0.202574 |

## SciFact

Metric: classification accuracy. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               182.840000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.700000 |           0.340000 | 0.480000 | 0.140000 |               400.940000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.700000 |           0.340000 | 0.480000 | 0.140000 |              4929.840000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Qwen     | PUBLIC_TEXT_READER | 0c5b120e284f0e5299dc | same-reader no evidence | 0.340000 | 0.160000 |  0.520000 |

## Statistical interpretation

Use paired 10,000-resample bootstrap intervals (seed 42): QASPER paper clusters, other datasets query/claim clusters. Saved comparisons include same-reader no-evidence and matched raw-passage controls; D also includes matched Mistral reader controls. A positive point estimate alone is insufficient for superiority. Intervals are nominal and exploratory; selecting best conditions on this cohort adds selection bias. No mixed raw QA/verification aggregate is valid.

Paired result files: hierarchy_pairwise_df.csv.

## Flags

- empty_evidence_strict_prefix: 297 logical rows.
- invalid_verification_label: 0 logical rows.
- generation_cap_reached: 0 logical rows.
- unmapped_gold_annotation: 147 logical rows.

Empty evidence follows the frozen strict-prefix budget rule; an oversized first unit is not silently skipped. Generation-cap hits require inspection, not deletion. Invalid SciFact outputs retain the predefined strict scoring. Gold-mapping gaps affect evidence-recall interpretation. Exact source support does not imply answer correctness.

Cost: 1,863 physical calls for 3,150 logical rows. Cache hits retain conceptual per-policy LM costs. Cached original timing fields are not measured end-to-end latency; that field remains null.

Oracles are retrospective ceilings and do not establish a deployable controller. Specialist/public-reader surfaces are separated in the saved reports.

## Complete condition summary

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens | dataset   |   prompt_tokens |   generated_tokens |   invalid_label_rate |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|:----------|----------------:|-------------------:|---------------------:|
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.487169 |           0.336919 | 0.640000 | 0.040000 |               864.260000 | HotpotQA  |      917.060000 |          11.120000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.437379 |           0.287129 | 0.540000 | 0.060000 |               201.720000 | HotpotQA  |      254.520000 |          11.720000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | HotpotQA  |       52.800000 |          13.820000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.404341 |           0.254091 | 0.520000 | 0.060000 |               427.860000 | HotpotQA  |      480.660000 |          13.040000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.419882 |           0.269633 | 0.600000 | 0.060000 |              1352.580000 | HotpotQA  |     1405.380000 |          12.940000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 | HotpotQA  |      218.700000 |          12.440000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.387303 |           0.237053 | 0.480000 | 0.080000 |               205.260000 | HotpotQA  |      258.060000 |          12.240000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.414020 |           0.263770 | 0.540000 | 0.060000 |               427.580000 | HotpotQA  |      480.380000 |          13.020000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 | HotpotQA  |      851.160000 |          11.980000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.380352 |           0.230102 | 0.440000 | 0.080000 |               205.460000 | HotpotQA  |      258.260000 |          12.400000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.475304 |           0.325054 | 0.620000 | 0.040000 |               868.060000 | HotpotQA  |      920.860000 |          11.620000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.391268 |           0.241019 | 0.460000 | 0.080000 |               842.880000 | HotpotQA  |      895.720000 |          12.460000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.346082 |           0.195833 | 0.420000 | 0.120000 |               406.420000 | HotpotQA  |      459.320000 |          13.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1350.500000 | HotpotQA  |     1403.300000 |          13.080000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 | HotpotQA  |      441.920000 |          12.140000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.410590 |           0.260340 | 0.580000 | 0.060000 |              1352.260000 | HotpotQA  |     1405.060000 |          13.180000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.403771 |           0.253522 | 0.580000 | 0.060000 |              1351.020000 | HotpotQA  |     1403.820000 |          13.020000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 | HotpotQA  |     1403.860000 |          13.520000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.442910 |           0.292660 | 0.560000 | 0.040000 |               426.060000 | HotpotQA  |      478.880000 |          12.040000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.287253 |           0.137003 | 0.340000 | 0.100000 |               193.240000 | HotpotQA  |      246.460000 |          14.340000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.479379 |           0.329129 | 0.620000 | 0.040000 |               867.640000 | HotpotQA  |      920.440000 |          11.380000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.182107 |           0.090879 | 0.580000 | 0.120000 |               864.500000 | QASPER    |      906.780000 |          19.540000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.182305 |           0.091077 | 0.500000 | 0.120000 |               204.260000 | QASPER    |      246.320000 |          19.200000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       41.980000 |          18.160000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.183501 |           0.092273 | 0.540000 | 0.160000 |               438.600000 | QASPER    |      480.740000 |          20.020000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.245822 |           0.154594 | 0.720000 | 0.080000 |              4146.420000 | QASPER    |     4188.580000 |          21.460000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 | QASPER    |       60.740000 |          17.960000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.103456 |           0.012229 | 0.080000 | 0.020000 |                23.440000 | QASPER    |       65.460000 |          18.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.196084 |           0.104857 | 0.580000 | 0.140000 |               435.920000 | QASPER    |      478.040000 |          20.180000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 | QASPER    |      816.520000 |          20.380000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.103456 |           0.012229 | 0.080000 | 0.020000 |                23.440000 | QASPER    |       65.460000 |          18.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.193604 |           0.102377 | 0.580000 | 0.160000 |               881.620000 | QASPER    |      923.720000 |          20.180000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.188630 |           0.097403 | 0.560000 | 0.100000 |               822.040000 | QASPER    |      864.220000 |          18.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.131073 |           0.039845 | 0.440000 | 0.120000 |               397.340000 | QASPER    |      439.580000 |          18.120000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 | QASPER    |     4188.820000 |          21.460000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 | QASPER    |      309.800000 |          19.320000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.800000 | QASPER    |     4188.960000 |          21.460000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.243726 |           0.152498 | 0.720000 | 0.060000 |              4146.780000 | QASPER    |     4188.940000 |          21.060000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 | QASPER    |     4186.440000 |          21.200000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.169549 |           0.078322 | 0.520000 | 0.120000 |               426.340000 | QASPER    |      468.520000 |          19.540000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.118661 |           0.027433 | 0.360000 | 0.180000 |               190.320000 | QASPER    |      232.700000 |          17.640000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.195769 |           0.104541 | 0.580000 | 0.160000 |               884.000000 | QASPER    |      926.080000 |          20.500000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.660000 |           0.300000 | 0.420000 | 0.120000 |               814.300000 | SciFact   |      872.640000 |           3.640000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               182.840000 | SciFact   |      241.200000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | SciFact   |       58.300000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.600000 |           0.240000 | 0.340000 | 0.100000 |               316.340000 | SciFact   |      374.640000 |           3.600000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.680000 |           0.320000 | 0.460000 | 0.140000 |              4928.660000 | SciFact   |     4986.960000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 | SciFact   |       77.660000 |           3.800000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                23.200000 | SciFact   |       81.500000 |           3.800000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.620000 |           0.260000 | 0.360000 | 0.100000 |               316.480000 | SciFact   |      374.780000 |           3.640000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 | SciFact   |      725.940000 |           3.680000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                23.200000 | SciFact   |       81.500000 |           3.800000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.580000 |           0.220000 | 0.380000 | 0.160000 |               820.860000 | SciFact   |      879.160000 |           3.400000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.460000 |           0.100000 | 0.280000 | 0.180000 |               779.080000 | SciFact   |      837.400000 |           3.920000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.460000 |           0.100000 | 0.320000 | 0.220000 |               377.380000 | SciFact   |      435.740000 |           4.200000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.700000 |           0.340000 | 0.480000 | 0.140000 |              4929.840000 | SciFact   |     4988.140000 |           3.760000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 | SciFact   |      296.000000 |           3.840000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.660000 |           0.300000 | 0.460000 | 0.160000 |              4931.520000 | SciFact   |     4989.820000 |           3.680000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.660000 |           0.300000 | 0.440000 | 0.140000 |              4942.700000 | SciFact   |     5001.000000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | FULL     | C0           |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 | SciFact   |     4994.740000 |           3.840000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 512      | C0           |     0.700000 |           0.340000 | 0.480000 | 0.140000 |               400.940000 | SciFact   |      459.280000 |           3.760000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 256      | C0           |     0.400000 |           0.040000 | 0.320000 | 0.280000 |               183.460000 | SciFact   |      241.840000 |           4.400000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER | HIERARCHICAL           | semantic   | 1024     | C0           |     0.620000 |           0.260000 | 0.380000 | 0.120000 |               819.280000 | SciFact   |      877.580000 |           3.400000 |             0.000000 |

## Post-hoc label-format diagnostic — not primary rescoring

SciFact strict scores require the whole stripped output to be one exact label. A label followed by an explanation is invalid under the frozen rule. Substantial format failures confound reader comparisons: low strict accuracy cannot be attributed solely to evidence utilization or reasoning. The separate anchored leading-label diagnostic accepts only an unambiguous label at the start, followed by a boundary/punctuation; it does not search explanations for labels. It was introduced after inspecting Mistral format failures and is exploratory. Primary scores, predictions, bootstrap comparisons and oracles remain unchanged. A prospective output-format protocol requires a new version.

| dataset   | reader   | action_id            | evidence_granularity   | selector   | budget   | compressor   |   n_queries |   strict_accuracy |   strict_format_rate |   leading_label_rate |   leading_label_diagnostic_accuracy |   cap_hit_rate |
|:----------|:---------|:---------------------|:-----------------------|:-----------|:---------|:-------------|------------:|------------------:|---------------------:|---------------------:|------------------------------------:|---------------:|
| SciFact   | Qwen     | 0962fbdcf76305eedb48 | HIERARCHICAL           | semantic   | 1024     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | 0c5b120e284f0e5299dc | HIERARCHICAL           | semantic   | 256      | C0           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | 0e8bebd7162b7ff88ee7 | NONE                   | none       | 0        | C0           |          50 |          0.360000 |             1.000000 |             1.000000 |                            0.360000 |       0.000000 |
| SciFact   | Qwen     | 1a6eb129284d5b2236bc | HIERARCHICAL           | semantic   | 512      | C0           |          50 |          0.600000 |             1.000000 |             1.000000 |                            0.600000 |       0.000000 |
| SciFact   | Qwen     | 217a31e1b7342ff3de03 | HIERARCHICAL           | semantic   | FULL     | C0           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | 239378f6325cdec057d8 | HIERARCHICAL           | semantic   | 256      | C0           |          50 |          0.380000 |             1.000000 |             1.000000 |                            0.380000 |       0.000000 |
| SciFact   | Qwen     | 2558b4cef17ee32423c9 | HIERARCHICAL           | semantic   | 256      | C0           |          50 |          0.380000 |             1.000000 |             1.000000 |                            0.380000 |       0.000000 |
| SciFact   | Qwen     | 2afd2c9a0f027158a914 | HIERARCHICAL           | semantic   | 512      | C0           |          50 |          0.620000 |             1.000000 |             1.000000 |                            0.620000 |       0.000000 |
| SciFact   | Qwen     | 2e32573f3f1d3e7e54b2 | HIERARCHICAL           | semantic   | 1024     | C0           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | 34f50f4577a44f718d6d | HIERARCHICAL           | semantic   | 256      | C0           |          50 |          0.380000 |             1.000000 |             1.000000 |                            0.380000 |       0.000000 |
| SciFact   | Qwen     | 5ff87da8d015f3790b49 | HIERARCHICAL           | semantic   | 1024     | C0           |          50 |          0.580000 |             1.000000 |             1.000000 |                            0.580000 |       0.000000 |
| SciFact   | Qwen     | 687776d5eb83efde75dc | HIERARCHICAL           | semantic   | 1024     | C0           |          50 |          0.460000 |             1.000000 |             1.000000 |                            0.460000 |       0.000000 |
| SciFact   | Qwen     | 6dfd0093cded4005430e | HIERARCHICAL           | semantic   | 512      | C0           |          50 |          0.460000 |             1.000000 |             1.000000 |                            0.460000 |       0.000000 |
| SciFact   | Qwen     | 7e58a4ce3a31ab304596 | HIERARCHICAL           | semantic   | FULL     | C0           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | b6d6a4f1235acd4d9740 | HIERARCHICAL           | semantic   | 512      | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | db48058d424931e66763 | HIERARCHICAL           | semantic   | FULL     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | db57ce79a22a0411beef | HIERARCHICAL           | semantic   | FULL     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | dfcceaf3326c2064d54a | HIERARCHICAL           | semantic   | FULL     | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | e4359fe1fbdcd0789ca0 | HIERARCHICAL           | semantic   | 512      | C0           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | ea42cfe81c7188941a89 | HIERARCHICAL           | semantic   | 256      | C0           |          50 |          0.400000 |             1.000000 |             1.000000 |                            0.400000 |       0.000000 |
| SciFact   | Qwen     | f6dab22d36c36e287281 | HIERARCHICAL           | semantic   | 1024     | C0           |          50 |          0.620000 |             1.000000 |             1.000000 |                            0.620000 |       0.000000 |

# Week 11 E — audited pilot results

4,150 available logical rows completed; 50 queries per dataset. Pending components: ['LLMLingua-2']. This is the frozen development pilot, not full-dataset confirmation.

Source/config/code identities, checksums, full action membership, budgets and same-reader/checkpoint HELP/HURT pairs passed. No scientific outputs were replaced.

C0 uses raw top-10-pool passages; C2 exposes up to eight exact query-ranked sentences with the same budgets. LLMLingua C1 remains pending. C3 is a QASPER/Mistral-only SARA specialist with its own adapted no-evidence baseline; it is not a portable text compressor and must be discussed separately. E has a different candidate pool width from C/D; compare compressor conditions within E for controlled conclusions.

## HotpotQA

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Mistral  | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.100453 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER | NONE                   | none       | 0        | C0           |     0.094824 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 |
| Llama    | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.424796 |           0.329972 | 0.700000 | 0.040000 |               762.960000 |
| Llama    | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 256      | C2           |     0.388219 |           0.293395 | 0.640000 | 0.080000 |               184.620000 |
| Mistral  | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 512      | C0           |     0.324281 |           0.223828 | 0.580000 | 0.140000 |               439.180000 |
| Mistral  | PUBLIC_TEXT_READER | PASSAGE                | semantic   | FULL     | C0           |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 |
| Mistral  | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.309067 |           0.208614 | 0.560000 | 0.200000 |               903.240000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |
| Qwen     | PUBLIC_TEXT_READER | PASSAGE                | semantic   | 512      | C0           |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   | 256      | C2           |     0.437931 |           0.287682 | 0.560000 | 0.040000 |               194.920000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Llama    | PUBLIC_TEXT_READER | 9198edbcf4344ac67bd5 | same-reader no evidence | 0.334868 | 0.233721 |  0.438197 |
| Mistral  | PUBLIC_TEXT_READER | 642251784cef20d8cbf0 | same-reader no evidence | 0.223828 | 0.132451 |  0.324101 |
| Qwen     | PUBLIC_TEXT_READER | c55c2cc46b508361dd08 | same-reader no evidence | 0.330022 | 0.230832 |  0.431392 |

## QASPER

Metric: answer F1. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant               | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-----------------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Mistral  | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Mistral  | MISTRAL_ONLY_LATENT_BASELINE | NONE                   | none       | 0        | C3           |     0.190914 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.071760 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.262209 |           0.190449 | 0.680000 | 0.100000 |              2351.320000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.198886 |           0.127126 | 0.580000 | 0.180000 |               759.700000 |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.184874 |           0.113115 | 0.560000 | 0.160000 |               195.920000 |
| Mistral  | MISTRAL_ONLY_LATENT_BASELINE | PASSAGE                | semantic   | FULL     | C3           |     0.403749 |           0.212834 | 0.500000 | 0.080000 |              1090.340000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.192379 |           0.132573 | 0.700000 | 0.100000 |               848.440000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.187197 |           0.127392 | 0.580000 | 0.160000 |              2632.440000 |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.185692 |           0.125887 | 0.620000 | 0.160000 |               228.600000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.218722 |           0.127494 | 0.660000 | 0.100000 |              2394.940000 |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.206276 |           0.115048 | 0.560000 | 0.180000 |               198.700000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant                      | action               | reference               |    point |   ci_low |   ci_high |
|:---------|:-----------------------------|:---------------------|:------------------------|---------:|---------:|----------:|
| Llama    | PUBLIC_TEXT_READER           | 9198edbcf4344ac67bd5 | same-reader no evidence | 0.190449 | 0.129025 |  0.255693 |
| Mistral  | MISTRAL_ONLY_LATENT_BASELINE | 4c4a8a869121679a3aa9 | same-reader no evidence | 0.212834 | 0.118354 |  0.304464 |
| Mistral  | PUBLIC_TEXT_READER           | 46c9eec30ae8d7fc6e33 | same-reader no evidence | 0.132573 | 0.094090 |  0.173052 |
| Qwen     | PUBLIC_TEXT_READER           | 86f6d0ff5e0e2c5ccfb8 | same-reader no evidence | 0.127494 | 0.088952 |  0.166841 |

## SciFact

Metric: classification accuracy. Baselines and top three non-baseline conditions per reader/checkpoint follow. Rankings are retrospective within this pilot.

| reader   | reader_variant     | evidence_granularity   | selector   |   budget | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:---------|:-------------------|:-----------------------|:-----------|---------:|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| Qwen     | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Mistral  | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER | NONE                   | none       |        0 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |
| Llama    | PUBLIC_TEXT_READER | PASSAGE                | semantic   |      256 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                19.140000 |
| Llama    | PUBLIC_TEXT_READER | SENTENCE               | semantic   |      256 | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               183.780000 |
| Llama    | PUBLIC_TEXT_READER | PASSAGE                | semantic   |      512 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               234.680000 |
| Mistral  | PUBLIC_TEXT_READER | PASSAGE                | semantic   |      256 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                23.440000 |
| Mistral  | PUBLIC_TEXT_READER | SENTENCE               | semantic   |      256 | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               226.640000 |
| Mistral  | PUBLIC_TEXT_READER | PASSAGE                | semantic   |      512 | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               285.200000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   |      256 | C2           |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               186.960000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   |      512 | C2           |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               342.240000 |
| Qwen     | PUBLIC_TEXT_READER | SENTENCE               | semantic   |     1024 | C2           |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               357.100000 |

Nominal paired intervals for each displayed reader/checkpoint best (selected retrospectively; not adjusted for selection):

| reader   | variant            | action               | reference                           |    point |   ci_low |   ci_high |
|:---------|:-------------------|:---------------------|:------------------------------------|---------:|---------:|----------:|
| Llama    | PUBLIC_TEXT_READER | 898eaba4b77f96c2090d | same-reader no evidence             | 0.000000 | 0.000000 |  0.000000 |
| Mistral  | PUBLIC_TEXT_READER | b4f75e721e90dbb239db | same-reader no evidence             | 0.000000 | 0.000000 |  0.000000 |
| Qwen     | PUBLIC_TEXT_READER | c0e284f6ff1027450ac4 | same-reader no evidence             | 0.340000 | 0.160000 |  0.520000 |
| Qwen     | PUBLIC_TEXT_READER | c0e284f6ff1027450ac4 | matched-budget raw semantic passage | 0.320000 | 0.140000 |  0.500000 |

## Statistical interpretation

Use paired 10,000-resample bootstrap intervals (seed 42): QASPER paper clusters, other datasets query/claim clusters. Saved comparisons include same-reader no-evidence and matched raw-passage controls; D also includes matched Mistral reader controls. A positive point estimate alone is insufficient for superiority. Intervals are nominal and exploratory; selecting best conditions on this cohort adds selection bias. No mixed raw QA/verification aggregate is valid.

Paired result files: compressor_pairwise_df.csv.

## Flags

- empty_evidence_strict_prefix: 297 logical rows.
- invalid_verification_label: 900 logical rows.
- generation_cap_reached: 214 logical rows.
- unmapped_gold_annotation: 203 logical rows.

Empty evidence follows the frozen strict-prefix budget rule; an oversized first unit is not silently skipped. Generation-cap hits require inspection, not deletion. Invalid SciFact outputs retain the predefined strict scoring. Gold-mapping gaps affect evidence-recall interpretation. Exact source support does not imply answer correctness.

Cost: 850 physical calls for 4,150 logical rows. Cache hits retain conceptual per-policy LM costs. Cached original timing fields are not measured end-to-end latency; that field remains null.

Oracles are retrospective ceilings and do not establish a deployable controller. Specialist/public-reader surfaces are separated in the saved reports.

## Complete condition summary

| reader   | reader_variant               | evidence_granularity   | selector   | budget   | compressor   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens | dataset   |   prompt_tokens |   generated_tokens |   invalid_label_rate |
|:---------|:-----------------------------|:-----------------------|:-----------|:---------|:-------------|-------------:|-------------------:|---------:|---------:|-------------------------:|:----------|----------------:|-------------------:|---------------------:|
| Qwen     | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | HotpotQA  |       52.800000 |          13.820000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.424796 |           0.329972 | 0.700000 | 0.040000 |               762.960000 | HotpotQA  |      838.400000 |          13.920000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 | HotpotQA  |      218.700000 |          12.440000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.100453 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | HotpotQA  |       53.360000 |          28.420000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.309067 |           0.208614 | 0.560000 | 0.200000 |               903.240000 | HotpotQA  |      957.060000 |          24.780000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.425640 |           0.275391 | 0.540000 | 0.060000 |               260.040000 | HotpotQA  |      312.840000 |          11.800000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.324281 |           0.223828 | 0.580000 | 0.140000 |               439.180000 | HotpotQA  |      493.060000 |          23.000000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.094824 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | HotpotQA  |       75.440000 |          18.780000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               248.920000 | HotpotQA  |      324.360000 |          15.680000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 | HotpotQA  |     1403.860000 |          13.520000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.254890 |           0.154437 | 0.620000 | 0.180000 |               300.020000 | HotpotQA  |      353.600000 |          30.160000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.284481 |           0.189657 | 0.580000 | 0.100000 |               157.560000 | HotpotQA  |      233.000000 |          15.920000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.254890 |           0.154437 | 0.620000 | 0.180000 |               300.020000 | HotpotQA  |      353.600000 |          30.160000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 | HotpotQA  |      441.920000 |          12.140000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.429692 |           0.334868 | 0.680000 | 0.060000 |              1293.580000 | HotpotQA  |     1369.020000 |          13.880000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.255692 |           0.155239 | 0.620000 | 0.180000 |               297.580000 | HotpotQA  |      351.160000 |          30.120000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.420640 |           0.270391 | 0.540000 | 0.060000 |               262.260000 | HotpotQA  |      315.060000 |          11.800000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               248.920000 | HotpotQA  |      324.360000 |          15.680000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.320750 |           0.220297 | 0.560000 | 0.220000 |              1527.300000 | HotpotQA  |     1581.120000 |          26.980000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.254523 |           0.154071 | 0.600000 | 0.200000 |               222.480000 | HotpotQA  |      276.100000 |          29.660000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.238869 |           0.138417 | 0.440000 | 0.240000 |               185.200000 | HotpotQA  |      239.020000 |          25.280000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.420640 |           0.270391 | 0.540000 | 0.060000 |               262.260000 | HotpotQA  |      315.060000 |          11.800000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.437931 |           0.287682 | 0.560000 | 0.040000 |               194.920000 | HotpotQA  |      247.720000 |          11.880000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 | HotpotQA  |      851.160000 |          11.980000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.386535 |           0.291711 | 0.660000 | 0.020000 |               370.380000 | HotpotQA  |      445.820000 |          13.860000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.354328 |           0.259504 | 0.680000 | 0.060000 |               246.920000 | HotpotQA  |      322.360000 |          15.680000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.388219 |           0.293395 | 0.640000 | 0.080000 |               184.620000 | HotpotQA  |      260.060000 |          14.940000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       41.980000 |          18.160000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.198886 |           0.127126 | 0.580000 | 0.180000 |               759.700000 | QASPER    |      825.280000 |          30.800000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 | QASPER    |       60.740000 |          17.960000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.059805 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       41.660000 |          22.040000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.192379 |           0.132573 | 0.700000 | 0.100000 |               848.440000 | QASPER    |      890.280000 |          41.020000 |           nan        |
| Mistral  | MISTRAL_ONLY_LATENT_BASELINE | PASSAGE                | semantic   | FULL     | C3           |     0.403749 |           0.212834 | 0.500000 | 0.080000 |              1090.340000 | QASPER    |     1215.140000 |          21.680000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.195583 |           0.104356 | 0.520000 | 0.180000 |               287.260000 | QASPER    |      329.320000 |          19.560000 |           nan        |
| Mistral  | MISTRAL_ONLY_LATENT_BASELINE | NONE                   | none       | 0        | C3           |     0.190914 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       70.660000 |          27.600000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.140820 |           0.081014 | 0.540000 | 0.180000 |               289.480000 | QASPER    |      331.260000 |          33.440000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.071760 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | QASPER    |       64.940000 |          17.720000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.177285 |           0.105525 | 0.540000 | 0.160000 |               287.780000 | QASPER    |      352.800000 |          22.420000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.218722 |           0.127494 | 0.660000 | 0.100000 |              2394.940000 | QASPER    |     2437.380000 |          21.500000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.179171 |           0.119366 | 0.600000 | 0.180000 |               334.740000 | QASPER    |      376.580000 |          34.120000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.085587 |           0.013827 | 0.040000 | 0.020000 |                18.560000 | QASPER    |       83.540000 |          17.900000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.179171 |           0.119366 | 0.600000 | 0.180000 |               334.740000 | QASPER    |      376.580000 |          34.120000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 | QASPER    |      309.800000 |          19.320000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.262209 |           0.190449 | 0.680000 | 0.100000 |              2351.320000 | QASPER    |     2416.720000 |          34.280000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.178653 |           0.118848 | 0.600000 | 0.180000 |               329.300000 | QASPER    |      371.140000 |          33.240000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.195984 |           0.104756 | 0.520000 | 0.180000 |               292.020000 | QASPER    |      334.080000 |          19.440000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.177285 |           0.105525 | 0.540000 | 0.160000 |               287.780000 | QASPER    |      352.800000 |          22.420000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.187197 |           0.127392 | 0.580000 | 0.160000 |              2632.440000 | QASPER    |     2674.240000 |          60.300000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.185692 |           0.125887 | 0.620000 | 0.160000 |               228.600000 | QASPER    |      270.440000 |          29.120000 |           nan        |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.064574 |           0.004769 | 0.060000 | 0.020000 |                19.660000 | QASPER    |       61.320000 |          22.500000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.195984 |           0.104756 | 0.520000 | 0.180000 |               292.020000 | QASPER    |      334.080000 |          19.440000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.206276 |           0.115048 | 0.560000 | 0.180000 |               198.700000 | QASPER    |      240.740000 |          20.180000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 | QASPER    |      816.520000 |          20.380000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.131191 |           0.059432 | 0.440000 | 0.200000 |               262.660000 | QASPER    |      328.320000 |          21.800000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.182285 |           0.110525 | 0.560000 | 0.160000 |               283.120000 | QASPER    |      348.140000 |          22.300000 |           nan        |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.184874 |           0.113115 | 0.560000 | 0.160000 |               195.920000 | QASPER    |      260.920000 |          20.220000 |           nan        |
| Qwen     | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | SciFact   |       58.300000 |           3.720000 |             0.000000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               653.920000 | SciFact   |      735.160000 |          99.740000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 | SciFact   |       77.660000 |           3.800000 |             0.000000 |
| Mistral  | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | SciFact   |       66.780000 |          79.860000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               800.320000 | SciFact   |      867.260000 |          70.680000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               342.240000 | SciFact   |      400.580000 |           3.560000 |             0.000000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               285.200000 | SciFact   |      351.980000 |          70.500000 |             1.000000 |
| Llama    | PUBLIC_TEXT_READER           | NONE                   | none       | 0        | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 | SciFact   |       81.220000 |         110.440000 |             1.000000 |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               349.340000 | SciFact   |      430.600000 |          97.660000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.640000 |           0.280000 | 0.420000 | 0.140000 |              3199.000000 | SciFact   |     3257.300000 |           3.640000 |             0.000000 |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               431.820000 | SciFact   |      498.780000 |          74.020000 |             1.000000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                19.140000 | SciFact   |      100.360000 |         106.500000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               431.820000 | SciFact   |      498.780000 |          74.020000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 | SciFact   |      296.000000 |           3.840000 |             0.000000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              3117.260000 | SciFact   |     3198.480000 |         112.760000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               414.880000 | SciFact   |      481.840000 |          74.920000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 1024     | C2           |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               357.100000 | SciFact   |      415.440000 |           3.560000 |             0.000000 |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               349.340000 | SciFact   |      430.600000 |          97.660000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | FULL     | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |              3832.180000 | SciFact   |     3899.100000 |          88.240000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               226.640000 | SciFact   |      293.440000 |          69.500000 |             1.000000 |
| Mistral  | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 256      | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |                23.440000 | SciFact   |       90.200000 |          78.320000 |             1.000000 |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | FULL     | C2           |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               357.100000 | SciFact   |      415.440000 |           3.560000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               186.960000 | SciFact   |      245.260000 |           3.720000 |             0.000000 |
| Qwen     | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 1024     | C0           |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 | SciFact   |      725.940000 |           3.680000 |             0.000000 |
| Llama    | PUBLIC_TEXT_READER           | PASSAGE                | semantic   | 512      | C0           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               234.680000 | SciFact   |      315.900000 |          86.740000 |             1.000000 |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 512      | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               335.580000 | SciFact   |      416.840000 |          97.160000 |             1.000000 |
| Llama    | PUBLIC_TEXT_READER           | SENTENCE               | semantic   | 256      | C2           |     0.000000 |           0.000000 | 0.000000 | 0.000000 |               183.780000 | SciFact   |      265.000000 |          88.640000 |             1.000000 |

## Post-hoc label-format diagnostic — not primary rescoring

SciFact strict scores require the whole stripped output to be one exact label. A label followed by an explanation is invalid under the frozen rule. Substantial format failures confound reader comparisons: low strict accuracy cannot be attributed solely to evidence utilization or reasoning. The separate anchored leading-label diagnostic accepts only an unambiguous label at the start, followed by a boundary/punctuation; it does not search explanations for labels. It was introduced after inspecting Mistral format failures and is exploratory. Primary scores, predictions, bootstrap comparisons and oracles remain unchanged. A prospective output-format protocol requires a new version.

| dataset   | reader   | action_id            | evidence_granularity   | selector   | budget   | compressor   |   n_queries |   strict_accuracy |   strict_format_rate |   leading_label_rate |   leading_label_diagnostic_accuracy |   cap_hit_rate |
|:----------|:---------|:---------------------|:-----------------------|:-----------|:---------|:-------------|------------:|------------------:|---------------------:|---------------------:|------------------------------------:|---------------:|
| SciFact   | Llama    | 1deae2b35b247dee530e | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.420000 |       0.260000 |
| SciFact   | Llama    | 752b8164fdeac6db93be | NONE                   | none       | 0        | C0           |          50 |          0.000000 |             0.000000 |             0.300000 |                            0.100000 |       0.540000 |
| SciFact   | Llama    | 7ca786528ae81d064555 | SENTENCE               | semantic   | 1024     | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.440000 |       0.260000 |
| SciFact   | Llama    | 898eaba4b77f96c2090d | PASSAGE                | semantic   | 256      | C0           |          50 |          0.000000 |             0.000000 |             0.380000 |                            0.120000 |       0.460000 |
| SciFact   | Llama    | 9198edbcf4344ac67bd5 | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.380000 |       0.580000 |
| SciFact   | Llama    | aecb54b081b08e1566b9 | SENTENCE               | semantic   | FULL     | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.440000 |       0.260000 |
| SciFact   | Llama    | da53b8dc3f3ebf165d43 | PASSAGE                | semantic   | 512      | C0           |          50 |          0.000000 |             0.000000 |             0.860000 |                            0.440000 |       0.180000 |
| SciFact   | Llama    | da8399ce3620653e1092 | SENTENCE               | semantic   | 512      | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.440000 |       0.260000 |
| SciFact   | Llama    | f682171aac01d7038bfa | SENTENCE               | semantic   | 256      | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.440000 |       0.120000 |
| SciFact   | Mistral  | 44e401eeab48a0c4bcb3 | NONE                   | none       | 0        | C0           |          50 |          0.000000 |             0.000000 |             0.920000 |                            0.440000 |       0.160000 |
| SciFact   | Mistral  | 46c9eec30ae8d7fc6e33 | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.000000 |             0.000000 |             0.980000 |                            0.680000 |       0.040000 |
| SciFact   | Mistral  | 642251784cef20d8cbf0 | PASSAGE                | semantic   | 512      | C0           |          50 |          0.000000 |             0.000000 |             0.980000 |                            0.640000 |       0.100000 |
| SciFact   | Mistral  | 88fa04e2bba3665564d6 | SENTENCE               | semantic   | FULL     | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.700000 |       0.120000 |
| SciFact   | Mistral  | 8f4cfef66be1182e8bc8 | SENTENCE               | semantic   | 1024     | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.700000 |       0.120000 |
| SciFact   | Mistral  | a69728887f07e1133fa5 | SENTENCE               | semantic   | 512      | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.680000 |       0.120000 |
| SciFact   | Mistral  | b25ba8af53973cc179f0 | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.000000 |             0.000000 |             0.980000 |                            0.600000 |       0.240000 |
| SciFact   | Mistral  | b276cd7bae341c426412 | SENTENCE               | semantic   | 256      | C2           |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.700000 |       0.020000 |
| SciFact   | Mistral  | b4f75e721e90dbb239db | PASSAGE                | semantic   | 256      | C0           |          50 |          0.000000 |             0.000000 |             0.920000 |                            0.400000 |       0.140000 |
| SciFact   | Qwen     | 0c72f8e9023a1e73f775 | NONE                   | none       | 0        | C0           |          50 |          0.360000 |             1.000000 |             1.000000 |                            0.360000 |       0.000000 |
| SciFact   | Qwen     | 4043cb2bb162608c9705 | PASSAGE                | semantic   | 256      | C0           |          50 |          0.380000 |             1.000000 |             1.000000 |                            0.380000 |       0.000000 |
| SciFact   | Qwen     | 5224d9230045c6bfc305 | SENTENCE               | semantic   | 512      | C2           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | 86f6d0ff5e0e2c5ccfb8 | PASSAGE                | semantic   | FULL     | C0           |          50 |          0.640000 |             1.000000 |             1.000000 |                            0.640000 |       0.000000 |
| SciFact   | Qwen     | 9166ab9da1ecb39d539b | PASSAGE                | semantic   | 512      | C0           |          50 |          0.660000 |             1.000000 |             1.000000 |                            0.660000 |       0.000000 |
| SciFact   | Qwen     | ae874a989b8c57264be3 | SENTENCE               | semantic   | 1024     | C2           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | b8c645f32688d98645d1 | SENTENCE               | semantic   | FULL     | C2           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| SciFact   | Qwen     | c0e284f6ff1027450ac4 | SENTENCE               | semantic   | 256      | C2           |          50 |          0.700000 |             1.000000 |             1.000000 |                            0.700000 |       0.000000 |
| SciFact   | Qwen     | c55c2cc46b508361dd08 | PASSAGE                | semantic   | 1024     | C0           |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |

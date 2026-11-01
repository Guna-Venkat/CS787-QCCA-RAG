# Week 11 C — granularity and budget pilot

All 5,550 executable conditions finished: 50 queries per dataset, 37 Qwen text conditions per query. The 600 configured token-compression conditions remain pending LLMLingua-2 qualification. This is a pilot, not full-dataset confirmation.

## Observations

QASPER benefits most from the larger context in this surface; HotpotQA peaks at a 1,024-token passage condition; SciFact has competitive small sentence budgets. These are within-pilot point estimates, not a universal winner. QA F1 and verification accuracy are kept separate.

## HotpotQA

No-evidence F1: **0.150250**.

Top point estimates (ties retain saved order):

| evidence_granularity   | selector   |   budget |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:-----------------------|:-----------|---------:|-------------:|-------------------:|---------:|---------:|-------------------------:|
| PASSAGE                | semantic   |     1024 |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |
| PROPOSITION            | semantic   |     1024 |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |
| SENTENCE               | semantic   |     1024 |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |
| SENTENCE               | hybrid     |     1024 |     0.470987 |           0.320737 | 0.600000 | 0.040000 |               865.600000 |
| PASSAGE                | semantic   |      512 |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |
| PASSAGE                | hybrid     |     1024 |     0.447304 |           0.297054 | 0.540000 | 0.060000 |               825.180000 |

Best displayed action versus no evidence: delta 0.330022, nominal paired 95% CI [0.230832, 0.431392]. Bootstrap: 10,000 resamples, seed 42; paper clusters for QASPER, query/claim clusters otherwise. The displayed best was found on these same 50 queries; its interval does not correct for winner selection or multiple comparisons.

## QASPER

No-evidence F1: **0.091228**.

Top point estimates (ties retain saved order):

| evidence_granularity   | selector   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:-----------------------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|
| PASSAGE                | hybrid     | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |
| PASSAGE                | semantic   | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |
| PASSAGE                | lexical    | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |
| PROPOSITION            | semantic   | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |
| PROPOSITION            | lexical    | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |
| PROPOSITION            | hybrid     | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |

Best displayed action versus no evidence: delta 0.152425, nominal paired 95% CI [0.106875, 0.197373]. Bootstrap: 10,000 resamples, seed 42; paper clusters for QASPER, query/claim clusters otherwise. The displayed best was found on these same 50 queries; its interval does not correct for winner selection or multiple comparisons.

## SciFact

No-evidence accuracy: **0.360000**.

Top point estimates (ties retain saved order):

| evidence_granularity   | selector   |   budget |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |
|:-----------------------|:-----------|---------:|-------------:|-------------------:|---------:|---------:|-------------------------:|
| PASSAGE                | hybrid     |     1024 |     0.780000 |           0.420000 | 0.500000 | 0.080000 |               630.720000 |
| PASSAGE                | lexical    |     1024 |     0.740000 |           0.380000 | 0.520000 | 0.140000 |               652.960000 |
| SENTENCE               | hybrid     |      512 |     0.720000 |           0.360000 | 0.460000 | 0.100000 |               397.100000 |
| SENTENCE               | hybrid     |      256 |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 |
| PROPOSITION            | lexical    |      512 |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               395.400000 |
| PROPOSITION            | lexical    |     1024 |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               810.360000 |

Best displayed action versus no evidence: delta 0.420000, nominal paired 95% CI [0.240000, 0.600000]. Bootstrap: 10,000 resamples, seed 42; paper clusters for QASPER, query/claim clusters otherwise. The displayed best was found on these same 50 queries; its interval does not correct for winner selection or multiple comparisons.

## Flags and limitations

- 308 intervention rows expose no evidence under the preregistered strict-prefix budget rule. These are preserved and itemized in `empty_evidence_audit.csv`; an oversized first passage can yield an empty block. This is a real limitation of the allocation policy.

- 0 invalid SciFact label rows; 0 rows reached the 128-token generation cap. These flags are observational; saved answers and scoring were preserved.

- Previously identified QASPER annotation-mapping gaps remain flagged. SciFact no-evidence claims have null retrieval recall; they still contribute to classification accuracy.

- FULL selectors can expose identical complete evidence and legitimately produce ties. Identical prompt reuse reduces physical calls; it does not establish independent replications.

- 3506 physical generation calls served 5,550 logical rows. Recorded selection/generation timings can be cached original measurements. End-to-end latency was not measured and no serving-latency claim is made.

- Exact source support validates exposure text, not answer correctness or semantic faithfulness. Gold annotations enter evaluation, not evidence ranking.

## Full result surface

| dataset   | evidence_granularity   | selector   | budget   |   task_score |   delta_task_score |     HELP |     HURT |   actual_evidence_tokens |   prompt_tokens |
|:----------|:-----------------------|:-----------|:---------|-------------:|-------------------:|---------:|---------:|-------------------------:|----------------:|
| HotpotQA  | NONE                   | none       | 0        |     0.150250 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       52.800000 |
| HotpotQA  | PASSAGE                | hybrid     | 1024     |     0.447304 |           0.297054 | 0.540000 | 0.060000 |               825.180000 |      877.980000 |
| HotpotQA  | PASSAGE                | hybrid     | 256      |     0.384781 |           0.234531 | 0.520000 | 0.040000 |               157.200000 |      210.020000 |
| HotpotQA  | PASSAGE                | hybrid     | 512      |     0.387291 |           0.237041 | 0.480000 | 0.060000 |               378.240000 |      431.060000 |
| HotpotQA  | PASSAGE                | hybrid     | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |
| HotpotQA  | PASSAGE                | lexical    | 1024     |     0.378796 |           0.228546 | 0.460000 | 0.060000 |               809.560000 |      862.360000 |
| HotpotQA  | PASSAGE                | lexical    | 256      |     0.367020 |           0.216770 | 0.480000 | 0.040000 |               145.900000 |      198.720000 |
| HotpotQA  | PASSAGE                | lexical    | 512      |     0.324755 |           0.174505 | 0.420000 | 0.100000 |               374.220000 |      427.020000 |
| HotpotQA  | PASSAGE                | lexical    | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |
| HotpotQA  | PASSAGE                | semantic   | 1024     |     0.480272 |           0.330022 | 0.620000 | 0.060000 |               798.360000 |      851.160000 |
| HotpotQA  | PASSAGE                | semantic   | 256      |     0.337510 |           0.187260 | 0.400000 | 0.120000 |               165.900000 |      218.700000 |
| HotpotQA  | PASSAGE                | semantic   | 512      |     0.456280 |           0.306030 | 0.600000 | 0.040000 |               389.100000 |      441.920000 |
| HotpotQA  | PASSAGE                | semantic   | FULL     |     0.431145 |           0.280895 | 0.580000 | 0.060000 |              1351.060000 |     1403.860000 |
| HotpotQA  | PROPOSITION            | hybrid     | 1024     |     0.442534 |           0.292285 | 0.560000 | 0.060000 |               864.560000 |      917.360000 |
| HotpotQA  | PROPOSITION            | hybrid     | 256      |     0.357872 |           0.207622 | 0.440000 | 0.060000 |               203.020000 |      255.840000 |
| HotpotQA  | PROPOSITION            | hybrid     | 512      |     0.377521 |           0.227271 | 0.520000 | 0.040000 |               423.000000 |      475.820000 |
| HotpotQA  | PROPOSITION            | hybrid     | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |
| HotpotQA  | PROPOSITION            | lexical    | 1024     |     0.433383 |           0.283134 | 0.600000 | 0.020000 |               865.480000 |      918.280000 |
| HotpotQA  | PROPOSITION            | lexical    | 256      |     0.316006 |           0.165757 | 0.360000 | 0.040000 |               199.800000 |      252.660000 |
| HotpotQA  | PROPOSITION            | lexical    | 512      |     0.316340 |           0.166090 | 0.380000 | 0.080000 |               424.380000 |      477.180000 |
| HotpotQA  | PROPOSITION            | lexical    | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |
| HotpotQA  | PROPOSITION            | semantic   | 1024     |     0.475548 |           0.325299 | 0.600000 | 0.040000 |               865.040000 |      917.840000 |
| HotpotQA  | PROPOSITION            | semantic   | 256      |     0.407943 |           0.257694 | 0.520000 | 0.080000 |               198.300000 |      251.100000 |
| HotpotQA  | PROPOSITION            | semantic   | 512      |     0.430585 |           0.280335 | 0.520000 | 0.060000 |               426.660000 |      479.460000 |
| HotpotQA  | PROPOSITION            | semantic   | FULL     |     0.418010 |           0.267761 | 0.580000 | 0.060000 |              1352.660000 |     1405.460000 |
| HotpotQA  | SENTENCE               | hybrid     | 1024     |     0.470987 |           0.320737 | 0.600000 | 0.040000 |               865.600000 |      918.400000 |
| HotpotQA  | SENTENCE               | hybrid     | 256      |     0.362142 |           0.211892 | 0.460000 | 0.060000 |               205.120000 |      257.920000 |
| HotpotQA  | SENTENCE               | hybrid     | 512      |     0.396624 |           0.246374 | 0.500000 | 0.060000 |               419.780000 |      472.600000 |
| HotpotQA  | SENTENCE               | hybrid     | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |
| HotpotQA  | SENTENCE               | lexical    | 1024     |     0.399654 |           0.249405 | 0.520000 | 0.060000 |               870.440000 |      923.240000 |
| HotpotQA  | SENTENCE               | lexical    | 256      |     0.348744 |           0.198495 | 0.380000 | 0.020000 |               200.580000 |      253.420000 |
| HotpotQA  | SENTENCE               | lexical    | 512      |     0.337007 |           0.186758 | 0.400000 | 0.060000 |               425.340000 |      478.140000 |
| HotpotQA  | SENTENCE               | lexical    | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |
| HotpotQA  | SENTENCE               | semantic   | 1024     |     0.472093 |           0.321844 | 0.620000 | 0.040000 |               866.740000 |      919.540000 |
| HotpotQA  | SENTENCE               | semantic   | 256      |     0.421265 |           0.271015 | 0.540000 | 0.060000 |               201.160000 |      253.960000 |
| HotpotQA  | SENTENCE               | semantic   | 512      |     0.420340 |           0.270090 | 0.560000 | 0.060000 |               423.360000 |      476.160000 |
| HotpotQA  | SENTENCE               | semantic   | FULL     |     0.405590 |           0.255340 | 0.580000 | 0.060000 |              1351.020000 |     1403.820000 |
| QASPER    | NONE                   | none       | 0        |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |
| QASPER    | PASSAGE                | hybrid     | 1024     |     0.212562 |           0.121334 | 0.640000 | 0.060000 |               769.600000 |      812.160000 |
| QASPER    | PASSAGE                | hybrid     | 256      |     0.091228 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       41.980000 |
| QASPER    | PASSAGE                | hybrid     | 512      |     0.171876 |           0.080648 | 0.460000 | 0.120000 |               267.660000 |      310.300000 |
| QASPER    | PASSAGE                | hybrid     | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |
| QASPER    | PASSAGE                | lexical    | 1024     |     0.221855 |           0.130627 | 0.640000 | 0.060000 |               769.600000 |      812.240000 |
| QASPER    | PASSAGE                | lexical    | 256      |     0.086783 |          -0.004444 | 0.000000 | 0.020000 |                 3.400000 |       45.380000 |
| QASPER    | PASSAGE                | lexical    | 512      |     0.158010 |           0.066782 | 0.440000 | 0.180000 |               262.400000 |      305.120000 |
| QASPER    | PASSAGE                | lexical    | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |
| QASPER    | PASSAGE                | semantic   | 1024     |     0.199183 |           0.107956 | 0.600000 | 0.100000 |               773.900000 |      816.520000 |
| QASPER    | PASSAGE                | semantic   | 256      |     0.096925 |           0.005698 | 0.060000 | 0.020000 |                18.720000 |       60.740000 |
| QASPER    | PASSAGE                | semantic   | 512      |     0.158710 |           0.067482 | 0.460000 | 0.180000 |               267.100000 |      309.800000 |
| QASPER    | PASSAGE                | semantic   | FULL     |     0.243652 |           0.152425 | 0.740000 | 0.060000 |              4144.280000 |     4186.440000 |
| QASPER    | PROPOSITION            | hybrid     | 1024     |     0.202180 |           0.110952 | 0.600000 | 0.200000 |               870.100000 |      912.140000 |
| QASPER    | PROPOSITION            | hybrid     | 256      |     0.195327 |           0.104099 | 0.480000 | 0.220000 |               202.760000 |      244.800000 |
| QASPER    | PROPOSITION            | hybrid     | 512      |     0.199701 |           0.108474 | 0.520000 | 0.200000 |               424.420000 |      466.460000 |
| QASPER    | PROPOSITION            | hybrid     | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |
| QASPER    | PROPOSITION            | lexical    | 1024     |     0.204258 |           0.113031 | 0.560000 | 0.140000 |               863.120000 |      905.180000 |
| QASPER    | PROPOSITION            | lexical    | 256      |     0.149739 |           0.058511 | 0.400000 | 0.220000 |               203.380000 |      245.440000 |
| QASPER    | PROPOSITION            | lexical    | 512      |     0.168587 |           0.077359 | 0.500000 | 0.180000 |               421.040000 |      463.100000 |
| QASPER    | PROPOSITION            | lexical    | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |
| QASPER    | PROPOSITION            | semantic   | 1024     |     0.200596 |           0.109368 | 0.560000 | 0.140000 |               870.780000 |      912.860000 |
| QASPER    | PROPOSITION            | semantic   | 256      |     0.209378 |           0.118151 | 0.520000 | 0.160000 |               203.400000 |      245.460000 |
| QASPER    | PROPOSITION            | semantic   | 512      |     0.189278 |           0.098050 | 0.520000 | 0.180000 |               426.840000 |      468.920000 |
| QASPER    | PROPOSITION            | semantic   | FULL     |     0.242887 |           0.151659 | 0.720000 | 0.080000 |              4146.660000 |     4188.820000 |
| QASPER    | SENTENCE               | hybrid     | 1024     |     0.200471 |           0.109243 | 0.580000 | 0.180000 |               872.980000 |      915.000000 |
| QASPER    | SENTENCE               | hybrid     | 256      |     0.197049 |           0.105822 | 0.500000 | 0.200000 |               200.560000 |      242.600000 |
| QASPER    | SENTENCE               | hybrid     | 512      |     0.196996 |           0.105768 | 0.540000 | 0.200000 |               424.560000 |      466.620000 |
| QASPER    | SENTENCE               | hybrid     | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |
| QASPER    | SENTENCE               | lexical    | 1024     |     0.195360 |           0.104133 | 0.520000 | 0.140000 |               866.180000 |      908.240000 |
| QASPER    | SENTENCE               | lexical    | 256      |     0.152126 |           0.060898 | 0.420000 | 0.200000 |               204.760000 |      246.840000 |
| QASPER    | SENTENCE               | lexical    | 512      |     0.192098 |           0.100871 | 0.520000 | 0.180000 |               424.040000 |      466.100000 |
| QASPER    | SENTENCE               | lexical    | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |
| QASPER    | SENTENCE               | semantic   | 1024     |     0.204816 |           0.113589 | 0.560000 | 0.140000 |               872.960000 |      915.020000 |
| QASPER    | SENTENCE               | semantic   | 256      |     0.211284 |           0.120056 | 0.560000 | 0.160000 |               201.460000 |      243.500000 |
| QASPER    | SENTENCE               | semantic   | 512      |     0.191808 |           0.100580 | 0.560000 | 0.180000 |               427.160000 |      469.240000 |
| QASPER    | SENTENCE               | semantic   | FULL     |     0.235650 |           0.144422 | 0.700000 | 0.080000 |              4147.040000 |     4189.200000 |
| SciFact   | NONE                   | none       | 0        |     0.360000 |           0.000000 | 0.000000 | 0.000000 |                 0.000000 |       58.300000 |
| SciFact   | PASSAGE                | hybrid     | 1024     |     0.780000 |           0.420000 | 0.500000 | 0.080000 |               630.720000 |      689.020000 |
| SciFact   | PASSAGE                | hybrid     | 256      |     0.360000 |           0.000000 | 0.020000 | 0.020000 |                19.620000 |       77.920000 |
| SciFact   | PASSAGE                | hybrid     | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               240.760000 |      299.080000 |
| SciFact   | PASSAGE                | hybrid     | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |
| SciFact   | PASSAGE                | lexical    | 1024     |     0.740000 |           0.380000 | 0.520000 | 0.140000 |               652.960000 |      711.260000 |
| SciFact   | PASSAGE                | lexical    | 256      |     0.340000 |          -0.020000 | 0.000000 | 0.020000 |                13.440000 |       71.740000 |
| SciFact   | PASSAGE                | lexical    | 512      |     0.620000 |           0.260000 | 0.360000 | 0.100000 |               234.140000 |      292.460000 |
| SciFact   | PASSAGE                | lexical    | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |
| SciFact   | PASSAGE                | semantic   | 1024     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               667.620000 |      725.940000 |
| SciFact   | PASSAGE                | semantic   | 256      |     0.380000 |           0.020000 | 0.040000 | 0.020000 |                19.360000 |       77.660000 |
| SciFact   | PASSAGE                | semantic   | 512      |     0.660000 |           0.300000 | 0.400000 | 0.100000 |               237.700000 |      296.000000 |
| SciFact   | PASSAGE                | semantic   | FULL     |     0.660000 |           0.300000 | 0.420000 | 0.120000 |              4936.440000 |     4994.740000 |
| SciFact   | PROPOSITION            | hybrid     | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               813.880000 |      872.240000 |
| SciFact   | PROPOSITION            | hybrid     | 256      |     0.700000 |           0.340000 | 0.480000 | 0.140000 |               186.620000 |      244.920000 |
| SciFact   | PROPOSITION            | hybrid     | 512      |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               396.000000 |      454.320000 |
| SciFact   | PROPOSITION            | hybrid     | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |
| SciFact   | PROPOSITION            | lexical    | 1024     |     0.700000 |           0.340000 | 0.460000 | 0.120000 |               810.360000 |      868.720000 |
| SciFact   | PROPOSITION            | lexical    | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               188.080000 |      246.500000 |
| SciFact   | PROPOSITION            | lexical    | 512      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               395.400000 |      453.720000 |
| SciFact   | PROPOSITION            | lexical    | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |
| SciFact   | PROPOSITION            | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               819.320000 |      877.620000 |
| SciFact   | PROPOSITION            | semantic   | 256      |     0.660000 |           0.300000 | 0.420000 | 0.120000 |               188.540000 |      246.840000 |
| SciFact   | PROPOSITION            | semantic   | 512      |     0.620000 |           0.260000 | 0.400000 | 0.140000 |               392.560000 |      450.900000 |
| SciFact   | PROPOSITION            | semantic   | FULL     |     0.680000 |           0.320000 | 0.480000 | 0.160000 |              4927.640000 |     4985.940000 |
| SciFact   | SENTENCE               | hybrid     | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               810.660000 |      869.020000 |
| SciFact   | SENTENCE               | hybrid     | 256      |     0.720000 |           0.360000 | 0.480000 | 0.120000 |               188.460000 |      246.780000 |
| SciFact   | SENTENCE               | hybrid     | 512      |     0.720000 |           0.360000 | 0.460000 | 0.100000 |               397.100000 |      455.420000 |
| SciFact   | SENTENCE               | hybrid     | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |
| SciFact   | SENTENCE               | lexical    | 1024     |     0.680000 |           0.320000 | 0.460000 | 0.140000 |               812.440000 |      870.800000 |
| SciFact   | SENTENCE               | lexical    | 256      |     0.700000 |           0.340000 | 0.440000 | 0.100000 |               188.360000 |      246.700000 |
| SciFact   | SENTENCE               | lexical    | 512      |     0.680000 |           0.320000 | 0.440000 | 0.120000 |               392.820000 |      451.140000 |
| SciFact   | SENTENCE               | lexical    | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |
| SciFact   | SENTENCE               | semantic   | 1024     |     0.620000 |           0.260000 | 0.420000 | 0.160000 |               824.500000 |      882.800000 |
| SciFact   | SENTENCE               | semantic   | 256      |     0.680000 |           0.320000 | 0.420000 | 0.100000 |               187.620000 |      245.920000 |
| SciFact   | SENTENCE               | semantic   | 512      |     0.680000 |           0.320000 | 0.400000 | 0.080000 |               394.620000 |      452.960000 |
| SciFact   | SENTENCE               | semantic   | FULL     |     0.680000 |           0.320000 | 0.440000 | 0.120000 |              4943.660000 |     5001.960000 |

## Continuation

Proceed to D with SFR/P0 and all reviewed C textual selections fixed; vary only the reader. No single winning C condition is promoted, and token conditions are explicitly waived as pending. D must reuse C selection identities and same-reader baselines. Existing results and scientific configuration remain unchanged.

# W9-G-PRE — Saved-output action-space preflight

FIT only. Zero model generation and zero controller training. Oracles use observed FIT scores and are not deployable.

## Fixed saved actions

| action   |   n_queries |       f1 |   exact_match |     help |     hurt |   final_prompt_tokens |   total_query_prompt_tokens |   total_query_generated_tokens |   num_lm_calls |   evidence_tokens |
|:---------|------------:|---------:|--------------:|---------:|---------:|----------------------:|----------------------------:|-------------------------------:|---------------:|------------------:|
| D0       |        1746 | 0.385631 |      0.227950 | 0.000000 | 0.000000 |            940.544101 |                  940.544101 |                      16.654639 |       1.000000 |          0.000000 |
| F0       |        1746 | 0.390024 |      0.226231 | 0.216495 | 0.209049 |           1085.768614 |                 2026.312715 |                      39.949026 |       2.000000 |         98.332188 |
| F2-A     |        1746 | 0.389616 |      0.231386 | 0.021764 | 0.008591 |           1179.446163 |                 2119.990263 |                      33.200458 |       2.000000 |         98.332188 |

## Oracle spaces

| action_space   | oracle   |   n_queries |       f1 |   exact_match |   best_fixed_f1 |   headroom |     help |     hurt |   final_prompt_tokens |   total_query_prompt_tokens |   total_query_generated_tokens |   num_lm_calls |   evidence_tokens |
|:---------------|:---------|------------:|---------:|--------------:|----------------:|-----------:|---------:|---------:|----------------------:|----------------------------:|-------------------------------:|---------------:|------------------:|
| PRE-A          | exact    |        1746 | 0.393462 |      0.233677 |        0.389616 |   0.003846 | 0.021764 | 0.000000 |            946.764032 |                  968.778351 |                      17.608820 |       1.024055 |          2.294387 |
| PRE-A          | epsilon  |        1746 | 0.393449 |      0.233677 |        0.389616 |   0.003833 | 0.021764 | 0.000000 |            946.046392 |                  966.129439 |                      17.402062 |       1.021764 |          2.053837 |
| PRE-B          | exact    |        1746 | 0.453880 |      0.266323 |        0.390024 |   0.063856 | 0.216495 | 0.000000 |            973.217068 |                 1182.256014 |                      23.571019 |       1.222222 |         22.205040 |
| PRE-B          | epsilon  |        1746 | 0.453843 |      0.266323 |        0.390024 |   0.063819 | 0.216495 | 0.000000 |            972.325315 |                 1176.536082 |                      23.353379 |       1.216495 |         21.591065 |
| PRE-C          | exact    |        1746 | 0.454843 |      0.266896 |        0.390024 |   0.064819 | 0.221077 | 0.000000 |            975.954754 |                 1189.313860 |                      23.897480 |       1.227377 |         22.717068 |
| PRE-C          | epsilon  |        1746 | 0.454801 |      0.266896 |        0.390024 |   0.064777 | 0.221077 | 0.000000 |            974.806415 |                 1182.976518 |                      23.650630 |       1.221077 |         22.054410 |

Exact/epsilon preference: D0, then F2-A, then F0; epsilon=0.01. Epsilon chooses the first eligible action, not necessarily the highest score.

## Action distribution

| action_space   | oracle   | action   |   count |     rate |   help_count |   hurt_count |   neutral_count |
|:---------------|:---------|:---------|--------:|---------:|-------------:|-------------:|----------------:|
| PRE-A          | exact    | D0       |    1704 | 0.975945 |            0 |            0 |            1704 |
| PRE-A          | exact    | F2-A     |      42 | 0.024055 |           38 |            0 |               4 |
| PRE-A          | epsilon  | D0       |    1708 | 0.978236 |            0 |            0 |            1708 |
| PRE-A          | epsilon  | F2-A     |      38 | 0.021764 |           38 |            0 |               0 |
| PRE-B          | exact    | D0       |    1358 | 0.777778 |            0 |            0 |            1358 |
| PRE-B          | exact    | F0       |     388 | 0.222222 |          378 |            0 |              10 |
| PRE-B          | epsilon  | D0       |    1368 | 0.783505 |            0 |            0 |            1368 |
| PRE-B          | epsilon  | F0       |     378 | 0.216495 |          378 |            0 |               0 |
| PRE-C          | exact    | D0       |    1349 | 0.772623 |            0 |            0 |            1349 |
| PRE-C          | exact    | F0       |     366 | 0.209622 |          358 |            0 |               8 |
| PRE-C          | exact    | F2-A     |      31 | 0.017755 |           28 |            0 |               3 |
| PRE-C          | epsilon  | D0       |    1360 | 0.778923 |            0 |            0 |            1360 |
| PRE-C          | epsilon  | F0       |     357 | 0.204467 |          357 |            0 |               0 |
| PRE-C          | epsilon  | F2-A     |      29 | 0.016609 |           29 |            0 |               0 |

## Unique exact-best cases

| category              |   count |     rate |
|:----------------------|--------:|---------:|
| D0 uniquely best      |      15 | 0.008591 |
| F0 uniquely best      |     366 | 0.209622 |
| F2-A uniquely best    |      13 | 0.007446 |
| tie involving D0/F0   |       0 | 0.000000 |
| tie involving D0/F2-A |     358 | 0.205040 |
| tie involving F0/F2-A |      18 | 0.010309 |
| three-way tie         |     976 | 0.558992 |

## F2-A contribution

```json
{
  "f2_unique_exact_best_count": 13,
  "f2_unique_exact_best_rate": 0.0074455899198167235,
  "f2_only_epsilon_eligible_count": 9,
  "f2_only_epsilon_eligible_rate": 0.005154639175257732,
  "f2_same_strict_answer_as_d0_rate": 0.9438717067583047,
  "f2_same_case_whitespace_normalized_answer_as_d0_rate": 0.9438717067583047,
  "f2_equal_f1_to_d0_rate": 0.9673539518900344,
  "exact_extra_oracle_f1": 0.0009634178579209784,
  "epsilon_extra_oracle_f1": 0.0009579278590522988
}
```

## Research route screens

```json
{
  "G1": {
    "decision": "NO-GO",
    "epsilon_headroom": 0.0038332437934255337,
    "threshold": 0.005,
    "rule": "epsilon oracle(D0,F2-A) minus best fixed >= threshold"
  },
  "G2": {
    "decision": "GO",
    "epsilon_headroom": 0.06381925695899743,
    "verifier_advantage": 0.059986013165571894,
    "threshold": 0.005,
    "material_advantage_threshold": 0.005,
    "rule": "epsilon oracle(D0,F0) headroom >=.005 AND advantage over PRE-A headroom >=.005; descriptive screen, not significance"
  },
  "G3": {
    "decision": "NO-GO",
    "extra_exact_oracle_f1": 0.0009634178579209784,
    "unique_best_rate": 0.0074455899198167235,
    "threshold": 0.003,
    "rule": "Three-action exact oracle minus D0/F0 exact oracle >=.003. Unique-best fraction is descriptive; no outcome-tuned extra cutoff."
  },
  "recommended_route": "G2",
  "controller_training_started": false,
  "lm_calls": 0,
  "interpretation": "Retrospective action-space research recommendation, not evidence of a learnable/deployable verifier."
}
```

## Action targets

```json
{
  "n_queries": 1746,
  "counts": {
    "HELP": 378,
    "NEUTRAL": 1003,
    "HURT": 365
  },
  "rates": {
    "HELP": 0.21649484536082475,
    "NEUTRAL": 0.5744558991981672,
    "HURT": 0.20904925544100803
  },
  "accept_target_prevalence": 0.21649484536082475,
  "definition": "FIT target for THIS F0: delta=F0_F1-D0_F1; ACCEPT iff delta>.01; continuous delta retained. Labels never inference features."
}
```

## Cost limitation

Oracle costs count the chosen action only and exclude action search. A post-generation F0 verifier pays the F0 candidate cost even when it keeps D0. It is a quality/safety mechanism, not a compute-saving retrieval gate. Candidate-generation costs are recorded separately. No measured end-to-end latency is invented.

A high oracle is an optimistic ceiling; it does not establish that inference-available features can distinguish HELP from HURT. FIT was already used for E3/F development; later paper-grouped OOF evaluates the verifier conditionally on those frozen upstream actions, not an independent end-to-end confirmation. No DEV/INTERNAL-VAL/TEST data is used.

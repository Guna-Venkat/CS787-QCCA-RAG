# Partial Week 11 D progress and blockers

Updated 2026-10-28T14:43:10.052449+00:00. This is an incomplete reader surface: Mistral 5,550/5,550, Qwen 5,550/5,550 (reused C), Llama 0/5,550. E–I have not started.

## Runtime blocker and verified correction

Llama pre-load guard reported 10.4 GB free versus 20.4 GB required; no Llama rows generated. Canonical readers now run in owned spawned processes, joined on close; no weight, prompt, evidence, decoding, evaluation or core configuration changes. Two cached Mistral answers reproduced exactly, including prompt/generated token counts; a fresh Llama qualification answer succeeded. Both reader workers exited and GPU memory returned to about 291 MiB. Twelve operational tests passed. Existing failure logs and notebook outputs were archived; all 11,100 completed row checksums passed. Three extra operational LM calls are recorded separately.

## Scientific interpretation blocker: label compliance

Mistral has 1,849 invalid strict-format SciFact outputs out of 1,850 logical rows. Most begin with a label and continue with an explanation. This is a major confound in strict reader-accuracy comparisons; it is not evidence that Mistral cannot reason over the claims. Mistral also has 326 generation-cap hits across the completed surface; detailed counts follow. Original predictions, strict scores and primary comparison rules are preserved.

The following anchored leading-label extraction is a post-hoc, retrospective diagnostic only. It is not primary rescoring or a change to the scientific protocol. It accepts an exact leading label with a word boundary/punctuation and never searches explanations for a label. No inference-time selection uses gold or these diagnostic scores.

| reader   | evidence_granularity   | selector   |   budget |   n_queries |   strict_accuracy |   strict_format_rate |   leading_label_rate |   leading_label_diagnostic_accuracy |   cap_hit_rate |
|:---------|:-----------------------|:-----------|---------:|------------:|------------------:|---------------------:|---------------------:|------------------------------------:|---------------:|
| Mistral  | PASSAGE                | semantic   |     1024 |          50 |          0.000000 |             0.000000 |             0.980000 |                            0.680000 |       0.040000 |
| Mistral  | SENTENCE               | semantic   |      256 |          50 |          0.000000 |             0.000000 |             1.000000 |                            0.700000 |       0.000000 |
| Mistral  | NONE                   | none       |        0 |          50 |          0.000000 |             0.000000 |             0.920000 |                            0.440000 |       0.160000 |
| Qwen     | NONE                   | none       |        0 |          50 |          0.360000 |             1.000000 |             1.000000 |                            0.360000 |       0.000000 |
| Qwen     | SENTENCE               | semantic   |      256 |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |
| Qwen     | PASSAGE                | semantic   |     1024 |          50 |          0.680000 |             1.000000 |             1.000000 |                            0.680000 |       0.000000 |

## Row flags

| flag                           |   logical_rows |
|:-------------------------------|---------------:|
| Mistral_SciFact_invalid_format |           1849 |
| Mistral_SciFact_cap_hits       |            264 |
| Mistral_QASPER_cap_hits        |             58 |
| Mistral_HotpotQA_cap_hits      |              4 |

## Continuation

Resume only missing Llama answers, keeping exact C selections and all saved Mistral/Qwen results. Treat D verification reader rankings as compliance-confounded, and retain strict primary scoring. E–I may proceed after D completes its mechanical audit with this caveat explicitly recorded. Optional LLMLingua conditions remain pending. A prospective formatting/scoring change belongs in a new output version.

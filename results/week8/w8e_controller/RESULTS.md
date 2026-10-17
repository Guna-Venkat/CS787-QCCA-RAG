# W8-E frozen controller results — 18 October 2026

Controller construction used measured canonical-draft U1–U12 token diagnostics, gain-weighted logistic regression (B0-GW), and five paper-grouped FIT OOF folds. Gain weights were fixed at 1+5 max(0,best_delta). The original W8-B validation-only/random-fold gate results were not reused as FIT predictions.

The score and safety thresholds are chosen on FIT OOF realized downstream F1, not AUC or an oracle-assisted objective. Within 0.003 of the highest F1, lower cumulative prompt tokens are preferred, then cached-generation latency, then simplicity. Family configurations and the overall selected policy were saved before the controller validation confirmation. Previous INTERNAL-VAL inspection and the W8-D closeout are acknowledged; this is development confirmation, not an untouched final test.

**FIT-selected policy: E5-DraftSemantic.** The validation table does not change that selection.

The full-grid best FIT F1 is 0.405683; the selected point is 0.002964 below it. The 0.003 tolerance is applied once globally. Other family rows show their own cost-efficient FIT operating points.

| Policy | FIT OOF F1 | Validation F1 | Promotion rate | HURT / all queries | Mean cumulative prompt tokens |
|---|---:|---:|---:|---:|---:|
| E0 | 0.385631 | 0.389061 | 0.00% | 0.00% | 935.16 |
| E1 | 0.405454 | 0.412572 | 100.00% | 9.94% | 2068.39 |
| E2 | 0.404339 | 0.406122 | 62.57% | 8.77% | 1651.94 |
| E3 | 0.401424 | 0.401963 | 62.57% | 7.02% | 1675.19 |
| E5-DraftSemantic | 0.402719 | 0.400080 | 42.11% | 4.68% | 1416.11 |
| E5-U4-A | 0.401446 | 0.399381 | 48.54% | 3.51% | 1509.90 |

E0 returns k4; E1 always promotes DraftSemantic; E2 uses B0-GW + DraftSemantic; E3 uses B0-GW + frozen U4-A; E5 variants add predicted HELP/HURT vetoes. E4 was omitted according to the earlier W8-D closeout decision. There is no second-promotion experiment.

## Selected operating point

Configuration: `{"complexity": 4, "family": "E5", "gate_threshold": 0.3, "policy_name": "E5-DraftSemantic", "selector": "S3", "tau_help": 0.0, "tau_hurt": 0.1, "uses_attribution": false}`

Validation F1 0.400080; gain over k4 +0.011018; gain over k6 -0.001926. Promotion rate 42.11%. HELP among promotions 13.19%; HURT among promotions 11.11%. All-query HURT is 4.68%.

Exact mean final prompt tokens: 1017.59. Exact cumulative prompt tokens: 1416.11. Both include the actual chat template and special tokens; promoted queries count two generation prompts.

## Paired paper intervals for the FIT-selected policy

10,000 paper resamples of fixed validation predictions; query-weighted differences. These intervals do not include variation from retraining or correct for all prior development experiments.

- Versus always-promote DraftSemantic: F1 difference -0.012493, 95% interval [-0.025562, -0.000327].
- Versus k4: F1 difference +0.011018, 95% interval [-0.004866, +0.027197].
- Versus k6: F1 difference -0.001926, 95% interval [-0.022801, +0.019613].

The frozen policy uses 31.54% fewer cumulative prompt tokens than always-promote DraftSemantic and lowers all-query HURT by 5.26%, but its F1 is lower. The selected policy's paired F1 interval versus always promoting is entirely below zero; intervals versus k4 and k6 include zero. There is no confirmed quality gain over either static baseline.

Static k6 has F1 0.402005 and 1352.27 mean prompt tokens. The selected policy has +4.72% cumulative prompt cost versus k6 and lower point-estimate F1. The present result does not establish a better quality/cost operating point than k6. Smaller final prompts alone would miss the cost of the initial draft generation. These are confirmation results; they do not authorize switching the frozen policy to a validation winner.

## What the result establishes

Decisions use only measured draft features and model predictions. Selected outcomes are looked up after the action; harmful outcomes are retained. A STOP action uses the base answer, not an oracle choice between outputs. Query ID, paper ID, original passage index, semantic rank, BM25 rank, proposed identity and selected action are preserved in the saved decision/replay records.

The replay uses real saved deterministic single-promotion generations. It does not rerun SARA for every threshold combination. Base and chosen promotion generation latency are summed from measured generation records. This is explicitly an estimated generation latency, **not measured end-to-end deployment latency**: routing, gate and selector overhead are not invented or omitted under an end-to-end label. A complete runtime/compute dominance claim still needs a deployed benchmark.

The original W8-C selector and results are unchanged. Invalid synthetic controller tuning outputs, including archived copies, were removed. Missing real inputs raise errors; no synthetic fallback is available. TEST remains untouched. Re-running the completed notebook verifies source hashes and inspects the frozen run rather than silently retuning on validation.

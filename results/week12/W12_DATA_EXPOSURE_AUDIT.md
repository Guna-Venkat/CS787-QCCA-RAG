# Week12 data exposure audit — active v3

|   CONFIRM |   CONFIRM_groups |   TRAIN |   TRAIN_groups |   VAL |   VAL_groups | dataset   |
|----------:|-----------------:|--------:|---------------:|------:|-------------:|:----------|
|         0 |                0 |    2178 |            790 |  1005 |          281 | QASPER    |
|       200 |              200 |   90447 |          90439 |  7155 |         7155 | HotpotQA  |
|        50 |               43 |     531 |            325 |   177 |          157 | SciFact   |

## Official source queries

|       |   HotpotQA |   QASPER |   SciFact |
|:------|-----------:|---------:|----------:|
| TRAIN |      90447 |     2593 |       809 |
| VAL   |       7405 |     1005 |       300 |

QASPER official TEST: 416 papers in source metadata; 1,309 historical queries across 411 papers in saved W9 TEST membership. Official TEST payloads/labels were not opened. QASPER TEST is permanently closed. HotpotQA/SciFact official TEST scoring is not locally configured; use the frozen DEV-derived confirmation groups.

## Historical identity audit

|   week |   HotpotQA |   QASPER |   SciFact |
|-------:|-----------:|---------:|----------:|
|      0 |          0 |     2319 |         0 |
|     10 |          0 |      233 |         0 |
|     11 |         57 |       50 |        55 |
|      2 |          0 |      231 |         0 |
|      3 |          0 |      231 |         0 |
|      4 |          0 |      462 |         0 |
|      5 |          0 |     2319 |         0 |
|      6 |          0 |      573 |         0 |
|      7 |          0 |     2319 |         0 |
|      8 |          0 |     2088 |         0 |
|      9 |          0 |     9251 |         0 |

These are artifact-scoped identity counts, not pooled unique scientific questions: old QASPER numeric IDs may repeat across source splits. Exact artifact hashes and per-week IDs are in `qbgf/v3/protocol/exposure_manifest.json.gz`. Pilot IDs and paper/group membership are in `W12_SPLIT_FREEZE.json` (lossless tracked gzip copy). The two large W11 CSVs were additionally checked in `exposure_supplement.json`; they add no unknown IDs. Remaining large surfaces are QASPER only; the historical source-paper union is excluded conservatively. Unavailable/off-machine history cannot be certified.

## Exclusions and freshness

858 QASPER source TRAIN papers are historically exposed; only 39 unused queries on 30 papers remain. This fails the preregistered 200-paper requirement. **QASPER_FRESH_CONFIRMATION_AVAILABLE = FALSE**. No fake confirmation is created. TRAIN excludes 98 historical validation papers, leaving 2,178 queries; official validation supplies 1,005 queries, with the stated development-only limitation.

HotpotQA removes exposed/pilot query groups from VAL/CONFIRM; normalized duplicate questions remain together. Shared provided corpus documents do not imply label exposure and are not claimed document-disjoint. SciFact uses connected cited-document groups across TRAIN/DEV, excluding exposed and held-out groups from gradients (531 eligible TRAIN, not an arbitrary cap).

All three cross-role query-ID, question-hash and grouping overlaps are zero. W11 50-query cohorts are HISTORICAL_DEVELOPMENT / HYPOTHESIS_GENERATION only. No confirmation labels feed selection, checkpointing or gradients; no confirmation outcome was evaluated. All eligible TRAIN enters the final dataset-balanced sampling pool. Source revisions are exact content SHA-256; split IDs, seed, timestamp, hash and creation commit are immutable. Changing the split requires a new output version.

See `W12_CORRECTED_PROTOCOL_REPORT.md` for training, tests, limitations and manual run instructions.

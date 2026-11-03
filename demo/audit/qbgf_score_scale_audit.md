# LiveDemo QBGF score-scale audit

**Confirmed:** before this corrective pass, `demo/live/engine.py` passed raw BM25 passage and unit scores into the same `src/week12/algorithm.options` softmax with temperature multiplier 5 used for cosine scores. Numerical stability from subtracting the maximum does not fix this scale incompatibility.

## Actual World Cup query distributions

| Native score source | n | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|---:|
| BM25_passages | 10 | 6.093437 | 12.355153 | 9.139188 | 2.022125 |
| BM25_units | 43 | 0.401488 | 13.698774 | 5.420748 | 3.134753 |
| SFR_passages | 10 | 0.648569 | 0.765326 | 0.697365 | 0.038800 |
| SFR_units | 37 | 0.461252 | 0.774396 | 0.635147 | 0.069988 |

These are actual pinned SFR scores and project BM25 scores from the requested World Cup input/query **using corrected live segmentation**. The unit-score values audit the former native scoring formula on that corrected inventory; they do not reconstruct unavailable pre-fix browser-session scores. Passage scores are independent of the segmentation change. Passage values cover each native retriever's own top 10; unit values cover the evidence-unit corpus inside that pool. The pools differ, so this table is not a matched score-distribution test. It demonstrates the scale mismatch at the old transform input.

## Exact passage softmax example

BM25's largest passage weight after softmax(5×score) is **98.77%**, versus **13.78%** for SFR. This is a query-specific example, not a universal calibration result.

## Sentence/unit normalization

QBGF normalizes unit scores **within each parent passage**, not globally across all units. The machine-readable distribution file also includes a globally normalized diagnostic maximum for each unit corpus; that field is descriptive and is not a QBGF per-passage weight. Actual within-passage examples follow.

| Pipeline | Passage | Number of units | Largest within-passage weight |
|---|---|---:|---:|
| BM25_units | P1 | 4 | 1.000000 |
| BM25_units | P2 | 5 | 0.938115 |
| BM25_units | P3 | 6 | 1.000000 |
| BM25_units | P5 | 2 | 0.999724 |
| BM25_units | P6 | 3 | 0.986462 |
| BM25_units | P7 | 4 | 1.000000 |
| BM25_units | P8 | 4 | 1.000000 |
| BM25_units | P9 | 3 | 0.999991 |
| BM25_units | P13 | 6 | 1.000000 |
| BM25_units | P14 | 6 | 0.999051 |
| SFR_units | P1 | 4 | 0.385703 |
| SFR_units | P2 | 5 | 0.211084 |
| SFR_units | P5 | 2 | 0.543279 |
| SFR_units | P6 | 3 | 0.387235 |
| SFR_units | P7 | 4 | 0.316835 |
| SFR_units | P8 | 4 | 0.362180 |
| SFR_units | P9 | 3 | 0.425191 |
| SFR_units | P10 | 2 | 0.741385 |
| SFR_units | P11 | 9 | 0.203671 |
| SFR_units | P15 | 1 | 1.000000 |

## Correction and scope

- New BM25-native QBGF live execution is **blocked**, not silently normalized. Pipeline-native BM25 remains available for compatible lexical selection methods.
- Controlled mode changes BM25/SFR candidate retrieval, then rescales neither source arbitrarily: it recomputes pinned **common SFR query–passage and query–unit cosine relevance** for both pools before invoking the unchanged allocator.
- Native SFR QBGF retains its SFR scoring path. Saved curated BM25/QBGF illustrations are retained byte-for-byte and visibly labelled as lexical-score pipeline-native variants, not controlled comparisons.
- The original canonical `src/week12/laboratory.py` can also supply lexical relevance for a BM25 variant; it was not modified. This audit does not assume that every implemented research sensitivity arm was executed or reported. Main frozen SFR results retain their historical identity.
- Representative scores, exact values, source IDs and texts are in `world_cup_correctness_v2/score_distributions.json`. No BM25-native QBGF answer was generated during this corrective pass.

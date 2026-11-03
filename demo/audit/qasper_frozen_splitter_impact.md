# Read-only historical QASPER splitter impact audit

QASPER used the frozen fallback splitter, not native sentence offsets. Week-11 reconstructs source papers from the official TRAIN Arrow file using the saved pilot paper indices; Week-12 uses all frozen VAL papers from official validation. Both source constructors remove `BIBREF` identifiers before segmentation. No answer was regenerated and no splitter/result was modified.

## Exact source-population census

| Source population | Papers | Passages | Frozen units | Citation-aware counterfactual segments | Affected frozen units | Affected % |
|---|---:|---:|---:|---:|---:|---:|
| W11 granularity pilot | 40 | 784 | 7066 | 7067 | 0 | 0.0000% |
| W12 frozen VAL | 281 | 5522 | 50274 | 50298 | 52 | 0.1034% |

Affected means an existing frozen unit subdivides under the explicit citation/Markdown LiveDemo adapter. This is a counterfactual boundary diagnostic, not proof of changed answer quality. Paragraph/chunk-edge effects, abbreviations and ordinary NLP ambiguity remain. All passages count once per unique paper, not once per query.

## Saved FIXED_SENTENCE/QBGF context coverage

| Method | Budget | Available queries | Selected units | Sentence occurrences | Resolved | Affected | Unresolved | Whole passages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FIXED_SENTENCE | 256 | 50 | 467 | 467 | 467 | 1 | 0 | 0 |
| FIXED_SENTENCE | 1024 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| QBGF_FULL | 256 | 50 | 511 | 511 | 511 | 1 | 0 | 0 |
| QBGF_FULL | 1024 | 50 | 1225 | 1155 | 1155 | 1 | 0 | 70 |

The full C/D prepared/answer shards are no longer present. The retained attribution export is a subset, so it cannot certify every selected unit across the 1005-query QASPER validation population. Unresolved IDs, if any, remain unknown rather than unaffected. Complete-passage QBGF selections naturally contain multiple sentences and are not counted as faulty sentence units.

## Native boundaries in the other datasets

`src/week11/datasets.py` builds HotpotQA offsets from the provided sentence arrays and SciFact offsets from each abstract sentence array; document titles are also explicit native units. `src/week11/hierarchy.py` prioritizes these native offsets. The fallback citation-regex mechanism therefore does not define these native sentence boundaries. This does not certify that every dataset-native unit is a perfect grammatical sentence.

## Interpretation and limitation

Frozen research results retain their original preprocessing exactly. Any nonzero impact is a documented limitation of the fallback unit definition, not permission to repair historical answers or infer a new method-quality improvement. The impact counts do not measure downstream score sensitivity. If observed selected-unit coverage is small, do not generalize it to absent historical contexts.

Full affected texts, offsets, unit IDs and counterfactual segments are saved in the adjacent JSON. The audit accessed source text and saved identity metadata only; it did not access QASPER TEST or run inference.

## Crucial distinction: adapter differences versus real citation merges

The broad adapter is not a grammatical oracle. Its **52 subdivisions (0.1034%) must not be called 52 historical sentence errors**. Many are LaTeX lines interpreted as Markdown structures or leading citation fragments split by masking. These demonstrate why the custom Live adapter must not replace academic benchmark preprocessing.

A direct source-text check finds **3 terminal-citation merge cases among 50,274 W12 frozen units (0.00597%)**, and **0 among 7,066 W11 pilot units**. The retained FIXED_SENTENCE/QBGF selected-context subset contains **zero of these three direct merge cases**. Its one broadly affected unit in each available condition is a leading `[7]) ...` fragment, not multiple grammatical sentences. Full historical selected contexts remain unavailable, so zero in the retained subset is not zero over all 1005 questions.

Exact affected citation examples, IDs and offsets are in `terminal_citation_merge_cases` in the JSON. The source has only three passages containing the audited adjacent terminal-citation pattern. This supports a rare observed source limitation, not a claim of material answer-score damage or proof of no damage. No frozen experiment is silently corrected.

## Final-freeze clarification (no new census or inference)

W11 direct terminal-citation merges observed: **0 / 7,066**. W12: **3 / 50,274 (0.00597%)**. None of those three appeared in retained selected contexts. Complete historical selections are unavailable; zero downstream impact cannot be proven. This audit does not establish material score damage.

The W11 counterfactual total differs by one (7,067 versus 7,066), while isolated frozen-unit resplitting flags zero. The totals are produced by whole-passage segmentation, whereas the affected statistic applies the adapter to each old unit separately. Context and citation/boundary ownership can change whole-passage counts without subdividing any isolated old unit. They are not required to add arithmetically. The +1 was not treated as a confirmed grammatical merge, and the freeze pass did not reopen or rerun QASPER.

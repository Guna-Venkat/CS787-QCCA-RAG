> **Historical audit note:** the source/Git status below describes the pre-freeze corrective audit. The final user-approved release now selectively versions the demo and its immutable archive. See [current reproducibility status](demo_reproducibility_status.md) and [FINAL_FREEZE_STATUS.md](../FINAL_FREEZE_STATUS.md). No historical numeric audit result was changed.

# Final Evidence Lab correctness and methodology audit

Date: 3 November 2026 · Branch: `gagan`

## Verdict

**The attached audits identified genuine methodological and presentation problems.** The previous native BM25/SFR selector comparison changed both retrieval and downstream relevance scoring, so it was not a pure retriever effect. Raw BM25 relevance also entered a softmax parameterization designed around cosine relevance, creating a serious scale mismatch. Citation-heavy custom inputs exposed a limitation of the shared frozen fallback splitter. Fine-span and demo-policy labels overstated or obscured what was being shown.

These corrections apply to the demo and its documentation. No frozen algorithm, benchmark output, validation/confirmation result, curated saved answer or method manifest was changed. Research tables were not recalculated. No new evidence algorithm was introduced.

## 1. Issue-by-issue corrections

| Issue | Previous behavior / verdict | Exact code path | Corrective action and before/after | Verification | Frozen impact |
|---|---|---|---|---|
| Retriever-effect confound | Genuine methodological misinterpretation if labelled retrieval-only: BM25 used lexical fine scores, SFR used cosine | `demo/live/engine.py`: retrieval and fine scoring | Separate Pipeline-native comparison from Controlled retriever-only comparison. Controlled BM25/SFR top-10 pools both receive common pinned SFR passage/unit scores. The same allocator and reader follow | Tests verify common scores, identical protocol invariants, and BM25 used only for candidate ranking; six real paired diagnostics passed | None |
| Raw BM25 → QBGF softmax | Genuine LiveDemo score-scale incompatibility | Engine scores → unchanged `src/week12/algorithm.options` → `softmax(tau=5)` | New BM25-native QBGF execution now fails explicitly; UI disables that path. Controlled BM25 candidates are rescored with SFR before QBGF. Saved lexical QBGF illustrations remain with a warning | Runtime rejection test, direct allocation-input checks and measured distribution audit | Historical code and saved values retained |
| Citation-boundary merging | Genuine segmentation limitation, already shared with frozen fallback; not just grouped display | `src/week11/hierarchy.build` → `src/week9/senttopk_e3/sentences.spans` | Before 39 World Cup units, 16 merged. Citation/Markdown Live adapter yields 63 source units. Exact historical fallback remains an explicit Researcher option | Exact P6=3, P9=3, P7=4 regression checks; offsets, citations, abbreviations, decimals and Markdown tests | Original splitter unchanged |
| Grammar/type labels | Misleading UI when generic fragments were called sentences; serializer grouping was not the root cause | `demo/live/segmentation.py`, `demo/components.py`, live UI | Typed evidence-unit badges and separate cards; QBGF human labels use actual selected evidence-unit counts instead of interpreting SENT_k as grammatical guarantees | Known merged units are labelled MULTI-SENTENCE SEGMENT; structural cards show SOURCE/HEADING/LIST SEGMENT | Curated saved units unchanged |
| Stale contexts and results | Prior cache keys gained segmentation identity in the first patch but did not include the new scorer protocol | `demo/live/protocol.py`, engine input/preparation keys, UI reuse guard | Keys now include correctness protocol version, segmentation, scoring mode and scorer identity; backend device is separately keyed. Legacy records are retained/archived with an obsolete note and refused for reuse | Cross-mode cache and migration tests; real diagnostic resume used all saved rows without generation | No old result files deleted |
| Fine-span terminology | “Clause-level” implies parsing beyond punctuation rules | Curated UI, shared cards, historical replay display, guide | Visible name is Fine-grained source spans; description is Deterministic punctuation-based subdivisions of exact source sentences. Internal PROPOSITION ID retained | Label tests; source extractor untouched | None |
| Demo-only arm ambiguity | Three illustrative policies lacked visible status | `demo/curated/ui.py` | PROPOSITION, MULTI_SENTENCE_2 and MULTI_SENTENCE_4 carry DEMO-ONLY REPRESENTATION in controls/cards and scope table | Label assertions; bundle checksum unchanged | None |
| Result provenance ambiguity | Custom, curated and frozen displays were not consistently badged | Live result protocol; curated local view; historical replay / research panel | LIVE_CUSTOM → LIVE CUSTOM RUN; CURATED_ILLUSTRATION → CURATED EXAMPLE; FROZEN_RESEARCH → FROZEN RESEARCH RESULT | Protocol tests plus existing full-app regressions; metadata added in views, not persisted into frozen rows | Numeric findings unchanged |
| Lack of controlled answer comparison | Ranking-only comparison could not show answer propagation | `demo/live/ui.py` | Keep Compare Retrievers; add requested Compare Retrievers End-to-End with a visible invariant panel, both pools, exact evidence, types, costs and answers. No custom winner/score | Synthetic Streamlit end-to-end interaction and real 12-condition diagnostic | None |
| Reproducibility | Entire demo was ignored by Git, so a push would not preserve it | Root `/demo/` rule | Policy retained; file/hash inventory, qualified dependency snapshot and a submission preservation recommendation generated | `git ls-files -- demo` returned zero files | Ignore policy unchanged |

## 2. Explicit answers to the thirteen audit questions

1. **Was the original World Cup segmentation wrong?** Yes for a grammatical sentence-level custom-text demonstration. The frozen fallback merged independent cited statements. This was problem A (boundary detection), not merely problem B (individual sentences grouped by the serializer). Grouping selected units by parent passage for tokenization did not turn a merged source unit into that error.
2. **What fraction was affected?** 16 of 39 old source units, **41.03%**, on the exact 15-passage input. The corrected operational inventory is 63 units: 61 manually counted sentence statements plus two structural fragments. This percentage is a source-boundary diagnostic, not answer-score sensitivity.
3. **Is the correction research-frozen?** No. It is an explicitly versioned **LiveDemo-only citation/Markdown adapter** that delegates to the frozen splitter while preserving original source offsets. Frozen research preprocessing remains available unchanged in Researcher Mode.
4. **Did any frozen result change?** No. All 771 protected tracked research/source files plus curated durable identities in the baseline snapshot verified unchanged. No research scorer/answer was rerun.
5. **Does selecting BM25/SFR change more than retrieval?** In **Pipeline-native comparison**, yes: downstream lexical versus semantic scoring can differ. That mode is labelled end-to-end pipeline behavior. In **Controlled retriever-only comparison**, the downstream SFR scorer and every registered protocol setting remain the same. Only candidate retrieval output (membership/ranking) changes.
6. **Is a retriever-only mode available?** Yes. Both candidate pools receive common SFR query–passage and query–unit cosine scoring; identical segmentation, allocation, budget, serializer, reader, template/parser and decoding follow. The end-to-end view verifies invariant equality.
7. **Do raw BM25 scores enter the exponential transform anywhere?** They did in the previous LiveDemo and remain represented in immutable curated native-QBGF saved illustrations. The original canonical research implementation also still supports lexical variants; it was not edited. **No new corrected LiveDemo QBGF run accepts raw BM25 allocation scores.** The score-audit script evaluates the former scoring formula to document it, without generating native BM25-QBGF answers.
8. **Where/how is that native mode labelled?** Native BM25 is permitted for lexical compatible selection methods and explicitly called Pipeline-native comparison. QBGF with native BM25 is blocked with an actionable message. Saved curated BM25/QBGF illustrations display a lexical-score/native-variant warning, not a controlled-comparison claim.
9. **Are controlled pools scored identically downstream?** Yes: the same pinned SFR encoder/revision, query instruction, document role, pooling/cosine rule and FP16 configuration. The pool content itself can differ; equal algorithms do not imply equal scores, evidence or answers.
10. **Can an object called a sentence contain multiple grammatical sentences?** Historical compatibility type `SENTENCE` can still denote an imperfect fallback segment, and no deterministic heuristic certifies arbitrary grammar. Known merged/structural cases receive accurate Live display badges; the World Cup adapter inventory has no remaining units that subdivide under its own citation-aware rule. Selected units are shown individually, not grouped under a singular sentence label. Residual abbreviation/linguistic ambiguity is explicitly a limitation, not claimed solved by counting periods.
11. **Are demo-only methods identified?** Yes, in the method controls, answer cards and per-condition scope table. Internal IDs remain compatible with saved data.
12. **Were old World Cup runs invalidated?** New protocol/segmentation/scorer keys and reuse checks refuse pre-fix sentence/QBGF contexts. When legacy session records exist, they are retained in a labelled archive instead of silently deleted. No persisted original World Cup answer files were available to recover; the exact input was supplied by the user. This does not invalidate RAW_SKIP merely because segmentation changed. A historical-splitter run made deliberately after the patch remains a labelled reproduction, not a clean sentence-granularity claim.
13. **Were frozen hashes preserved?** Yes. `correctness_protected_before.json` records the baseline, and `protect_artifacts.py --verify` verifies all 771 files. The primary-source manifest and both final-method freeze identities also pass.

## 3. Exact World Cup corrective experiment

The requested **12 conditions completed** in 62.3 seconds on the available GPU, including scoring/model handoff. The app's default remains sequential CPU FP16. Six controlled pairs passed invariant equality. There were **11 distinct exact-prompt generations** for 12 logical answer rows; exact-prompt reuse is retained as one conceptual reader call per condition. A second execution resumed from all 12 immutable rows without loading large models or regenerating answers.

Query: Explain how the expansion of the 2026 World Cup changed both the group-stage structure and the total tournament workload, and explain why FIFA abandoned the originally planned three-team group format.

| Retriever | Method | Budget | Evidence tokens | Prompt tokens | Generated tokens |
|---|---|---:|---:|---:|---:|
| BM25 | FIXED_SENTENCE | 256 | 256 | 327 | 65 |
| BM25 | FIXED_SENTENCE | 1024 | 1024 | 1095 | 79 |
| BM25 | QBGF_FULL | 256 | 246 | 317 | 77 |
| BM25 | QBGF_FULL | 1024 | 1011 | 1082 | 64 |
| BM25 | RAW_SKIP | 256 | 169 | 240 | 61 |
| BM25 | RAW_SKIP | 1024 | 941 | 1012 | 60 |
| SFR | FIXED_SENTENCE | 256 | 256 | 327 | 65 |
| SFR | FIXED_SENTENCE | 1024 | 1020 | 1091 | 70 |
| SFR | QBGF_FULL | 256 | 251 | 322 | 63 |
| SFR | QBGF_FULL | 1024 | 1022 | 1093 | 82 |
| SFR | RAW_SKIP | 256 | 245 | 316 | 56 |
| SFR | RAW_SKIP | 1024 | 993 | 1064 | 82 |

The [complete 12-condition report](world_cup_correctness_v2/WORLD_CUP_12_CONDITION_REPORT.md) includes every answer and exact context. Each condition JSON contains ranked passage IDs/scores, selected IDs/text, source-unit types, exact tokens, per-unit selected/full-passage-covered status, QBGF option decisions and invariant metadata.

The eight requested concepts are traced through candidate-pool and selected-text phrase matches: 48-team expansion, 12 groups of four, old 16 groups of three, 64→104 matches, 7→8 semifinalist matches, 39 days, collusion and penalty shootouts. This is descriptive text inspection, not a gold metric or correctness judgment. No old/new answer difference is presented as an improvement in QBGF quality. The supplied source claims were not fact-checked.

## 4. QBGF score distributions

| Input score corpus | n | Min | Max | Mean | Std |
|---|---:|---:|---:|---:|---:|
| BM25_passages | 10 | 6.093437 | 12.355153 | 9.139188 | 2.022125 |
| BM25_units | 43 | 0.401488 | 13.698774 | 5.420748 | 3.134753 |
| SFR_passages | 10 | 0.648569 | 0.765326 | 0.697365 | 0.038800 |
| SFR_units | 37 | 0.461252 | 0.774396 | 0.635147 | 0.069988 |

These unit corpora use corrected custom segmentation and the former native scoring formula; they are not recovered pre-fix session values. Passage scores do not depend on that boundary change. The largest actual passage softmax weight was 98.77% for BM25 versus 13.78% for SFR on this query. Unit softmax is per passage; a global unit normalization is not the actual allocator weight. The [score-scale report](qbgf_score_scale_audit.md) provides exact distinctions and representative per-passage weights.

## 5. Read-only historical QASPER impact

QASPER did **not** supply native sentence offsets. The W11/W12 constructors flattened exact source material (removing BIBREF markers) and used the frozen bounded-passage plus academic-sentence fallback. The read-only census used exact source text and saved cohorts, not regenerated evidence embeddings or answers.

| Source | Papers | Passages | Frozen units | Counterfactual units | Affected | Affected % |
|---|---:|---:|---:|---:|---:|---:|
| W11 granularity cohort | 40 | 784 | 7066 | 7067 | 0 | 0.0000% |
| W12 complete frozen VAL | 281 | 5522 | 50274 | 50298 | 52 | 0.1034% |

The saved W11 cohort files use the same 50 QASPER questions/40 papers across the inspected phases. The detailed [QASPER report](qasper_frozen_splitter_impact.md) shows selected-context resolution/affected counts separately. Complete C/D prepared shards were removed before this audit, so full selected-context sensitivity across all 1005 VAL questions cannot be certified. Available retained attribution contexts are a subset; unknown context coverage is not reported as zero impact.

**Attribution of the boundary differences:** the 52 broad W12 subdivisions (0.1034%) are not 52 grammatical errors. They include Markdown-like equation lines and leading-citation artifacts introduced by the counterfactual live adapter. Direct source inspection identifies **3 actual terminal-citation merge cases (0.00597%)**, versus **0** in the W11 pilot. None of those three appears in the retained selected-context subset; its one broad adapter difference per available method/budget is a leading citation fragment. Full population answer sensitivity remains unknown.

HotpotQA and SciFact constructors use provided native sentence arrays (plus explicit title units), so the fallback regex does not determine their native sentence boundaries. This does not assert that every dataset-native unit is a grammatical sentence. Counterfactual boundary changes alone do not quantify answer-score changes. Historical results remain frozen, and the QASPER fallback definition/coverage limit is documented.

## 6. Reproducibility status

`git ls-files -- demo` returns **zero** tracked files. Source, tokenizer-only assets, immutable curated bundle, publication receipt and curated manifest are currently absent from a clean checkout. The root `/demo/` ignore rule was not changed.

Preserve source/small assets in version control after an explicitly approved final unfreeze; preserve the immutable bundle and checksum/manifest together in a versioned release or durable institutional data archive, with restore/verification instructions. Model weights and credentials should not be included. [Detailed inventory and submission plan](demo_reproducibility_status.md).

## 7. Tests and unchanged artifacts

- Complete demo suite: **138 passed in 36.49 seconds**.
- Controlled comparison tests verify identical downstream protocol and scorer, reject changed invariants, and confirm BM25 is called only for candidate retrieval in controlled QBGF.
- Native BM25-QBGF execution rejection, segmentation/scorer cache separation, legacy archive/non-reuse and live result typing are covered.
- Citation adjacency, multiple citations, letter references, headings/lists, abbreviations, decimals, exact offsets and P6/P9/P7 regressions are covered.
- Synthetic Streamlit testing clicks the end-to-end comparison and validates its invariant panel without real model weights.
- Missing historical replay handling remains safe; visible fine-span and demo-only labels are covered.
- 771 protected research/source/curated files verified unchanged.
- Live source diagnosis/12 custom conditions only: no benchmark validation, confirmation, training, TEST run or historical answer regeneration.

## 8. Files and manual use

Updated: `demo/live/engine.py`, `demo/live/ui.py`, new `demo/live/protocol.py`, `demo/curated/ui.py`, shared display components, historical display wording, guide and tests. Added read-only source/score/reproducibility audit utilities and custom diagnostic receipts under `demo/audit/`. No src/week* file was modified.

1. Restart Streamlit after the source update: `demo/.venv/bin/python -m streamlit run demo/app.py`. A browser refresh alone can retain old imported modules/resources.
2. In LiveDemo, keep Citation/Markdown-aware Live segmentation for pasted articles. Exact historical fallback is available only in Researcher Mode and has a warning.
3. Enable local models and choose Controlled retriever-only comparison when comparing BM25 and SFR downstream answers. Both need the common local SFR scorer in this mode.
4. Run a query, then Compare Retrievers End-to-End. Inspect the invariant panel, pools, typed source units, exact contexts and answers. Do not declare a custom answer winner without a reference.
5. Native lexical context exploration remains available, but new BM25-native QBGF is blocked. Curated results are unchanged, with scope/native-variant labels.

## Remaining limitations

- Rule segmentation is not a universal grammatical parser; unknown citation syntax, malformed web text and ambiguous abbreviation endings can still require inspection. The adapter is a live-only boundary policy.
- No pre-fix World Cup predictions were recovered from the inaccessible ephemeral browser session. The new diagnostic is reproducible from the supplied input and query; old predictions are not invented.
- Historical selected-context coverage is limited by retained exports. Source census and observed selected-unit counts do not establish downstream score sensitivity.
- The demo remains ignored by Git until the user approves a preservation-policy change. Do not treat a normal repository push as a backup of the application.
- No new algorithm was added; the final work is correction, explicit protocol separation and provenance.

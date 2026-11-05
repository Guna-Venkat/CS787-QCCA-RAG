from pathlib import Path
import json
import pandas as pd
root=Path(__file__).resolve().parents[2];base=root/'results/post_w12/bea_qbgf/v3';out=base/'pilot/aspect_mechanism_audit'
def read(name):return pd.read_csv(out/(name+'.csv'))
def table(x):return x.to_markdown(index=False,floatfmt='.4f')+'\n\n'
parts=['''# Saved-pilot audit: aspect decomposition, stopping and FULL escalation

**Scope:** existing 150-query VAL pilot only; 900 saved answer rows, 150 saved problem caches, and saved selections/traces. Comparisons use both FIXED_SENTENCE and QBGF_FULL at each budget. No model generation, embedding computation, parameter changes, tuning, expansion, or confirmation access.

## Findings

1. **The aspect concern is confirmed descriptively:** 49/50 QASPER queries have exactly one extracted aspect. Hotpot has 26/50 single-aspect queries; SciFact has 40/50. The means are 1.64 / 1.02 / 1.20. In this pilot, all extracted aspects happened to meet the represented-aspect cosine threshold, so the two reported means coincide. They remain different definitions.
2. **Single-aspect coverage is not zero or constant:** QASPER still gets later coverage rewards in every query. It is improving the best similarity to one query fragment, rather than covering distinct decomposed information needs. At 256, mean total weighted coverage reward is 0.3607, with 0.1264 after the first action; at 1024, 0.3640 total and 0.1298 later. Thus replacing the interpretation with exactly `Rel − lambda Red` would be inaccurate.
3. **The aspect heuristic is lexical, not semantic decomposition.** It splits punctuation/conjunctions, filters fragments shorter than two words, caps at six, and falls back to the whole query. A single fragment can contain multiple information needs; several fragments can be incomplete phrases rather than independent sub-questions. The examples below demonstrate both issues. No decomposition-quality gold labels exist here.
4. **QASPER losses are real but not explained by aspect count alone.** BEA loses 0.0799 / 0.0812 answer F1 against FIXED at 256 / 1024. There is only **one** multi-aspect QASPER query, so there is no adequate comparison group to estimate a decomposition effect. At 256, BEA already uses 96.0% of budget; large unused context cannot be the sole explanation for that cell.
5. **Correct the FULL denominator:** Hotpot has **14 upgrade events in 13/50 queries** at 256 and **25 events in 19/50 queries** at 1024. QASPER has none; SciFact has 0/50 and 2/50. The hypothesis that escalation is mostly a Hotpot phenomenon is supported in this pilot.
6. **Lower redundancy does not establish downstream utility:** every QASPER query has lower measured redundancy than FIXED, yet mean F1 is lower. At QASPER 1024, unused tokens correlate negatively with relative F1 (Spearman −0.308 vs FIXED, −0.259 vs QBGF), which is consistent with possible under-allocation but does not prove that adding tokens would help.

The existing preregistered gate remains **PROMISING**: it allows up to two clearly degraded cells, and both QASPER cells are flagged. This audit does not rewrite that decision. It supports pausing automatic expansion for a mechanism discussion; it does not authorize changing aspects, thresholds or methods after these outcomes.

## Definitions and analysis limits

- **Extracted aspect count:** `len(problem.aspects)`, one value per unique query; distributions below use 50 queries/dataset, not 100 budget rows.
- **Represented aspects:** count of extracted fragments whose maximum cosine to exposed sentences is at least 0.5. That threshold is diagnostic only; the allocator's coverage reward uses continuous positive increases in maximum similarity, without this 0.5 cutoff.
- **Coverage is cosine, not verified fact coverage.** A whole scientific question can encode several requirements even when the lexical splitter returns one fragment. No causal utility claim follows from cosine scores.
- **Win/loss/tie:** QA compares saved per-query F1 using a numerical tolerance of 1e-12. SciFact compares saved per-claim correctness, because macro-F1 is not a per-claim score. Exact wins differ from the project's separate >0.01 QA HELP/HURT convention; both labels are retained in the paired CSV.
- The main aggregate table uses QA answer F1 and SciFact three-class macro-F1. Relationship tables use **QA F1 difference / SciFact accuracy difference**; this keeps per-query associations interpretable. Full CSVs additionally contain correctly recomputed subset macro-F1. Those subset values are class-composition dependent, nonadditive and unstable for tiny strata; a correctness tie can still change macro-F1 through a different wrong class.
- Relationships are exploratory descriptions of the same selected pilot, not new confirmation, causal effects or independent evidence. No correlation p-values, new multiplicity family or tuning choices are introduced. Each budget uses the same underlying 50 queries. Do not pool budgets as independent observations or pool task metrics.

## 1. Extracted aspect-count distributions

''']
parts.append(table(read('aspect_distribution')))
o=read('mechanism_overview');parts.append('### Extracted versus represented counts and actual allocation\n\n')
parts.append(table(o[['dataset','budget','mean_extracted_aspects','mean_represented_aspects','one_aspect_queries','mean_unused_tokens','threshold_stops']]))
parts.append('## 2. Aggregate quality and wins/losses\n\nDelta is BEA minus the named comparator; all cells contain 50 queries. The SciFact primary delta is macro-F1; its win/loss labels use correctness.\n\n')
summary=pd.read_csv(base/'pilot/summary.csv');pairs=read('paired_query_diagnostics');agg=[]
for (d,b,c),xs in pairs.groupby(['dataset','budget','comparator']):
 def quality(m):return summary[(summary.dataset==d)&(summary.budget==b)&(summary.method==m)].iloc[0].quality
 agg.append(dict(dataset=d,budget=b,comparator=c,primary_metric='macro-F1' if d=='SciFact' else 'answer F1',delta_primary=quality('BEA_QBGF')-quality(c),wins=sum(xs.outcome=='win'),losses=sum(xs.outcome=='loss'),ties=sum(xs.outcome=='tie')))
parts.append(table(pd.DataFrame(agg)))
parts.append('### Aspect counts among wins, losses and ties\n\nPrimary reference: FIXED_SENTENCE. The complete corresponding QBGF table and exact count distributions are in `outcome_mechanisms.csv` and `aspect_counts_by_outcome.csv`.\n\n')
x=read('outcome_mechanisms');parts.append(table(x[x.comparator=='FIXED_SENTENCE'][['dataset','budget','outcome','queries','mean_aspect_count','mean_unused_tokens','mean_redundancy_reduction','full_queries']]))
parts.append('## 3. Quality difference versus extracted aspect count\n\nThe `3+` Hotpot stratum contains seven queries; the two-aspect QASPER stratum contains **one**. A positive value in that lone case cannot establish a general benefit. Hotpot shows no stable monotonic count–quality pattern across both budgets/comparators.\n\n')
x=read('quality_by_aspect_group');parts.append(table(x[['dataset','budget','comparator','stratum','n','wins','losses','mean_query_delta']].rename(columns={'stratum':'aspect count','mean_query_delta':'delta F1 / accuracy'})))
parts.append('## 4. Quality difference versus unused tokens\n\nRanges below are unused fractions of the selected budget: [0,25%], (25%,50%], (50%,75%], (75%,100%]. These are descriptive bins set for this audit, not candidate policies. QASPER@1024 versus FIXED ranges from +0.0656 F1 for the nine queries using at least 75% of budget to −0.2597 for the eight leaving more than 75% unused. Input difficulty, query type and evidence quality can confound this pattern.\n\n')
x=read('quality_by_unused_group');parts.append(table(x[x.comparator=='FIXED_SENTENCE'][['dataset','budget','stratum','n','wins','losses','mean_query_delta']].rename(columns={'stratum':'unused fraction','mean_query_delta':'delta F1 / accuracy'})))
parts.append('## 5. Quality difference versus redundancy reduction\n\nReduction = comparator pairwise sentence cosine minus BEA pairwise cosine. Positive means BEA has lower measured redundancy. FULL is measured using its constituent sentences; undefined values are excluded from correlations rather than converted to zero. All current paired values are defined. Reduced redundancy is not a guarantee of preserving complementary evidence, discourse context or answer utility.\n\n')
x=read('quality_by_redundancy_group');parts.append(table(x[x.comparator=='FIXED_SENTENCE'][['dataset','budget','stratum','n','wins','losses','mean_query_delta']].rename(columns={'mean_query_delta':'delta F1 / accuracy'})))
parts.append('### Rank associations for both comparators\n\nSpearman rho uses tied ranks within each dataset/budget/comparator, n=50. Blank entries mean a constant feature (e.g. no FULL activity), not zero correlation. No association here identifies the cause of losses.\n\n')
x=read('correlations');features=['aspect_count','unused_tokens','redundancy_reduction','full_active'];x=x[x.feature.isin(features)]
t=x.pivot(index=['dataset','budget','comparator'],columns='feature',values='spearman_rho').reindex(columns=features).reset_index();t.columns.name=None;parts.append(table(t))
parts.append('## 6. FULL activation and rejection\n\nEvery query reached eligibility at both budgets; the two-sentence prerequisite did not prevent **all consideration** on these saved trajectories. It could still affect when eligibility arises; there is no counterfactual relaxed-prerequisite run. Counts of rejection/choice below are repeated candidate-step events, **not distinct queries**.\n\n')
x=read('full_diagnostics');parts.append(table(x[['dataset','budget','eligible_queries','full_queries','chosen','rejected_cost','rejected_threshold','rejected_low_gain','lost_competition']]))
parts.append('QASPER@256: 3,013 cost rejections versus 12 threshold rejections. QASPER@1024: 7,027 threshold rejections, 231 cost rejections and 527 admissible candidates losing global competition. This supports a cost bottleneck at 256 and a gain-per-token/competition bottleneck at 1024. Counts revisit candidates as selection evolves; they are not independent trials.\n\n')
parts.append('### Quality conditioned on FULL activation\n\nAssociation only: FULL and non-FULL queries differ before treatment. Hotpot@256 FULL queries fare better relative to FIXED (+0.0615 F1, n=13) than non-FULL queries (−0.1141, n=37). That is **not** a same-query ablation showing that FULL caused an improvement. The active SciFact group has only two claims.\n\n')
x=read('quality_by_full_group');parts.append(table(x[['dataset','budget','comparator','stratum','n','wins','losses','mean_query_delta']].rename(columns={'mean_query_delta':'delta F1 / accuracy'})))
parts.append('## 7. Stop reasons for wins and losses\n\nCounts below include ties for complete denominators. `BUDGET_EXHAUSTED` means no remaining positive action fit; `LOW_MARGINAL_VALUE` means positive fitting actions existed but failed the threshold. Hotpot@1024 stops at threshold for **every** query, including wins and losses, so that stop reason alone cannot explain which queries lose. QASPER@256 has 12/17 losses ending for budget fit and 5/17 ending at threshold, again ruling out threshold stopping as the only explanation.\n\n')
parts.append(table(read('stop_reasons_by_outcome')))
parts.append('## 8. Ten saved aspect examples per dataset\n\nThe first ten query IDs in lexicographic order are shown without selecting by outcome. Exact aspect lists come from the saved problem cache. Original raw questions were not retained in these cache objects; we do **not** reconstruct them or load fresh dataset records. IDs identify the queries; the strings below are the exact cached fragments. The CSV preserves them in full. All 150 lists are in `aspect_inventory.csv`.\n\n')
ex=read('aspect_examples')
for d,xs in ex.groupby('dataset'):
 parts.append('### '+d+'\n\n')
 xs=xs.copy();xs['aspects']=xs['aspects'].str.replace('|',' / ',regex=False);parts.append(table(xs[['query_id','count','aspects']]))
parts.append('''## What this establishes before expansion

- The present QASPER run is operationally **query-fragment relevance/coverage + redundancy-aware sentence allocation + stopping**, with no FULL escalation. Describing it as empirically demonstrated multi-granular fusion on QASPER would overstate the saved evidence.
- It does **not** establish that absent semantic decomposition caused the losses: QASPER has almost no within-dataset aspect-count variation; many questions ask one attribute; and the 256 losses persist despite near-full budgets. The single-aspect reward still changes selection, so it is not equivalent to deleting coverage.
- At 1024 the under-allocation association warrants caution, while the tight-budget failures indicate that evidence content/ranking or the utility surrogate also needs explanation. These are hypotheses, not resolved causal findings.
- FULL is available but mostly unused outside Hotpot. Its query/event denominator and gain-per-token rejection behavior must be reported accurately. No inference that the prerequisite or threshold should be changed follows from this audit alone.
- **No tuning or expanded run was performed.** The historical final method and pilot freeze/gate remain intact. Do not present these post-hoc correlations as a new validation result or a successful no-coverage/no-FULL ablation.

## Integrity and saved artifacts

The audit validated 900 answer envelopes and checksum/identity pairing, 150 problem caches and 900 saved selections, reproduced all 18 existing primary-quality summary cells to tolerance 1e-12, and verified **1,954 saved input files remained byte-identical**. Source hashes and the script hash are in `INPUT_MANIFEST.json`. The audit imports no model, tokenizer, encoder or experiment runner. No benchmark file or scientific implementation was edited.

All CSVs and provenance: `results/post_w12/bea_qbgf/v3/pilot/aspect_mechanism_audit/`. The per-query CSV has **600 paired rows** (300 BEA query/budget actions × two comparators); the unique-query aspect distribution has 150 rows. Main displayed relationships use FIXED where specified; complete QBGF tables are retained for unused-token and redundancy strata as well.

Reproduce with:

```bash
sara_env/bin/python W12-POST/scripts/audit_saved_pilot.py
```

Branch: gagan. All analysis uses saved pilot data only.
''')
report=''.join(parts)
(root/'W12-POST/PILOT_ASPECT_MECHANISM_AUDIT.md').write_text(report)
(out/'ASPECT_MECHANISM_AUDIT.md').write_text(report)
print('Written audit report:',root/'W12-POST/PILOT_ASPECT_MECHANISM_AUDIT.md')

import json
from pathlib import Path
from collections import defaultdict
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.week12.algorithm import softmax
p=Path(__file__).parent;d=json.loads((p/'world_cup_correctness_v2/score_distributions.json').read_text())['distributions']
lines=['# LiveDemo QBGF score-scale audit','',
'**Confirmed:** before this corrective pass, `demo/live/engine.py` passed raw BM25 passage and unit scores into the same `src/week12/algorithm.options` softmax with temperature multiplier 5 used for cosine scores. Numerical stability from subtracting the maximum does not fix this scale incompatibility.','',
'## Actual World Cup query distributions','', '| Native score source | n | Min | Max | Mean | Std |','|---|---:|---:|---:|---:|---:|']
for name,v in d.items():
 s=v['stats'];lines.append(f"| {name} | {s['n']} | {s['min']:.6f} | {s['max']:.6f} | {s['mean']:.6f} | {s['std']:.6f} |")
lines+=['','These are actual pinned SFR scores and project BM25 scores from the requested World Cup input/query. Passage values cover each native retriever\'s own top 10; unit values cover the evidence-unit corpus inside that pool. The pools differ, so this table is not a matched score-distribution test. It demonstrates the scale mismatch at the old transform input.','',
'## Exact passage softmax example','',f"BM25's largest passage weight after softmax(5×score) is **{100*d['BM25_passages']['stats']['softmax_tau5_largest_weight']:.2f}%**, versus **{100*d['SFR_passages']['stats']['softmax_tau5_largest_weight']:.2f}%** for SFR. This is a query-specific example, not a universal calibration result.",'',
'## Sentence/unit normalization','',
'QBGF normalizes unit scores **within each parent passage**, not globally across all units. The machine-readable distribution file also includes a globally normalized diagnostic maximum for each unit corpus; that field is descriptive and is not a QBGF per-passage weight. Actual within-passage examples follow.','', '| Pipeline | Passage | Number of units | Largest within-passage weight |','|---|---|---:|---:|']
for name in ['BM25_units','SFR_units']:
 groups=defaultdict(list)
 for row in d[name]['values']:groups[row['passage']].append(row['score'])
 for pid,vs in groups.items():lines.append(f'| {name} | {pid} | {len(vs)} | {max(softmax(vs)):.6f} |')
lines+=['','## Correction and scope','',
'- New BM25-native QBGF live execution is **blocked**, not silently normalized. Pipeline-native BM25 remains available for compatible lexical selection methods.','- Controlled mode changes BM25/SFR candidate retrieval, then rescales neither source arbitrarily: it recomputes pinned **common SFR query–passage and query–unit cosine relevance** for both pools before invoking the unchanged allocator.','- Native SFR QBGF retains its SFR scoring path. Saved curated BM25/QBGF illustrations are retained byte-for-byte and visibly labelled as lexical-score pipeline-native variants, not controlled comparisons.','- The original canonical `src/week12/laboratory.py` can also supply lexical relevance for a BM25 variant; it was not modified. This audit does not assume that every implemented research sensitivity arm was executed or reported. Main frozen SFR results retain their historical identity.','- Representative scores, exact values, source IDs and texts are in `world_cup_correctness_v2/score_distributions.json`. No BM25-native QBGF answer was generated during this corrective pass.']
(p/'qbgf_score_scale_audit.md').write_text('\n'.join(lines)+'\n')

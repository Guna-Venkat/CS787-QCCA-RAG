"""Attribute broad adapter differences; do not equate all subdivisions with errors."""
import json,csv,re
from pathlib import Path
out=Path(__file__).parent;root=out.parents[1]
x=json.loads((out/'qasper_frozen_splitter_impact.json').read_text())
pattern=re.compile(r'[.!?](?:\s*\[(?:\d+(?:\s*[,–—-]\s*\d+)*|[A-Z])\])+\s+\S')
for key in ('W11_pilot','W12_VAL'):
 matches=[r for r in x[key]['affected'] if pattern.search(r['text'])]
 x[key]['terminal_citation_merge_cases']=matches
 x[key]['counts']['terminal_citation_merge_units']=len(matches)
 x[key]['counts']['terminal_citation_merge_percent']=100*len(matches)/x[key]['counts']['frozen_units']
rows=list(csv.DictReader((root/'results/week12/attribution/v1/analysis/unit_importance.csv').open()))
true_ids={r['unit_id'] for r in x['W12_VAL']['terminal_citation_merge_cases']}
for row in x['selected_export_coverage']:
 selected=[r for r in rows if r['dataset']=='QASPER' and r['method']==row['method'] and int(float(r['budget']))==row['budget']]
 row['terminal_citation_merge_selected_occurrences']=sum(r['unit_id'] in true_ids for r in selected)
x['counterfactual_caution']='The 52 broad adapter subdivisions are not 52 grammatical errors. Most are equation/list boundaries or citation fragments introduced by the live-only adapter itself. Three exact terminal-citation merge cases occur in 50,274 W12 fallback units; none occurs in the retained selected-context subset. This is not a whole-population answer-sensitivity result.'
(out/'qasper_frozen_splitter_impact.json').write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
p=out/'qasper_frozen_splitter_impact.md';s=p.read_text();s+='\n## Crucial distinction: adapter differences versus real citation merges\n\nThe broad adapter is not a grammatical oracle. Its **52 subdivisions (0.1034%) must not be called 52 historical sentence errors**. Many are LaTeX lines interpreted as Markdown structures or leading citation fragments split by masking. These demonstrate why the custom Live adapter must not replace academic benchmark preprocessing.\n\nA direct source-text check finds **3 terminal-citation merge cases among 50,274 W12 frozen units (0.00597%)**, and **0 among 7,066 W11 pilot units**. The retained FIXED_SENTENCE/QBGF selected-context subset contains **zero of these three direct merge cases**. Its one broadly affected unit in each available condition is a leading `[7]) ...` fragment, not multiple grammatical sentences. Full historical selected contexts remain unavailable, so zero in the retained subset is not zero over all 1005 questions.\n\nExact affected citation examples, IDs and offsets are in `terminal_citation_merge_cases` in the JSON. The source has only three passages containing the audited adjacent terminal-citation pattern. This supports a rare observed source limitation, not a claim of material answer-score damage or proof of no damage. No frozen experiment is silently corrected.\n';p.write_text(s)
print('Refined historical citation cases:',{k:x[k]['counts']['terminal_citation_merge_units'] for k in ('W11_pilot','W12_VAL')})

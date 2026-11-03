"""Read-only fallback-boundary census. Reads QASPER TRAIN/VAL source text, no TEST or answers."""
import sys,json,re,csv,hashlib,time
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from datasets import Dataset
from demo.live.resources import LiveResources,configuration
from demo.live.segmentation import sentence_spans,FROZEN,ADAPTED
from src.week11.hierarchy import bounded_passages
from src.week11.cache import digest
out=Path(__file__).parent;cfg=configuration();tok=LiveResources().tokenizer();start=time.monotonic()
freeze=json.loads((ROOT/'results/week12/qbgf/v3/protocol/W12_SPLIT_FREEZE.json').read_text())
w11=json.loads((ROOT/'results/week11/multigranular/v1/granularity_budget/cohort.json').read_text())
pilot=sorted({int(x['document_id']) for x in w11['selected'] if x['dataset']=='QASPER'})
w12_ids={r['paper_id'] for r in freeze['datasets']['QASPER']['VAL']}
raw=Path(cfg['datasets']['QASPER']['raw_path']);assert 'test' not in raw.name.lower()

def source(p):
 sections=[p['title']+'\t'+p['abstract']]
 for name,paras in zip(p['full_text']['section_name'],p['full_text']['paragraphs']):
  if paras and paras!=['']:sections.append(re.sub(r'BIBREF\d+','',str(name)+'\t'+' '.join(paras)))
 return '\n\n'.join(sections)

def census(label,docs):
 totals=dict(papers=0,passages=0,frozen_units=0,counterfactual_segments=0,affected_units=0,passages_with_terminal_citation_pattern=0);affected=[];known={}
 for did,text in docs:
  totals['papers']+=1
  for a,b in bounded_passages(text,tok,cfg['hierarchy']['max_passage_tokens']):
   t=text[a:b];totals['passages']+=1
   totals['passages_with_terminal_citation_pattern']+=bool(re.search(r'[.!?]\[(?:\d+|[A-Z])',t))
   spans=sentence_spans(t,FROZEN);totals['frozen_units']+=len(spans);totals['counterfactual_segments']+=len(sentence_spans(t,ADAPTED))
   for x,y in spans:
    v=t[x:y];parts=sentence_spans(v,ADAPTED);uid=digest(['QASPER',did,'SENTENCE',a+x,a+y,v])[:32];known[uid]=len(parts)>1
    if len(parts)>1:
     totals['affected_units']+=1;affected.append(dict(document_id=did,unit_id=uid,start_char=a+x,end_char=a+y,text=v,counterfactual_parts=[v[c:d] for c,d in parts]))
  if totals['papers']%20==0:print(label,totals['papers'],'papers',round(time.monotonic()-start),'seconds',flush=True)
 totals['affected_percent']=100*totals['affected_units']/max(1,totals['frozen_units'])
 return dict(counts=totals,affected=affected),known

train=Dataset.from_file(str(raw)).select_columns(['id','title','abstract','full_text'])
a,known11=census('W11 pilot',[(str(i),source(train[i])) for i in pilot])
validation=Dataset.from_file(str(raw.with_name('qasper-validation.arrow'))).select_columns(['id','title','abstract','full_text'])
b,known12=census('W12 VAL',[(str(p['id']),source(p)) for p in validation if str(p['id']) in w12_ids])
# Audit frozen selected units only where the retained attribution export provides
# exact IDs. Do not invent selections removed during disk cleanup.
rows=list(csv.DictReader((ROOT/'results/week12/attribution/v1/analysis/unit_importance.csv').open()))
selected=[r for r in rows if r['dataset']=='QASPER' and r['method'] in ('FIXED_SENTENCE','QBGF_FULL')]
coverage=[]
for method in ('FIXED_SENTENCE','QBGF_FULL'):
 for budget in ('256','1024'):
  group=[r for r in selected if r['method']==method and r['budget'] in (budget,budget+'.0')]
  fine=[r for r in group if r['granularity']=='SENTENCE'];resolved=[r for r in fine if r['unit_id'] in known12]
  coverage.append(dict(method=method,budget=int(budget),available_queries=len({r['query_id'] for r in group}),selected_unit_occurrences=len(group),sentence_unit_occurrences=len(fine),resolved_sentence_occurrences=len(resolved),affected_sentence_occurrences=sum(known12[r['unit_id']] for r in resolved),unresolved_sentence_occurrences=len(fine)-len(resolved),whole_passage_occurrences=sum(r['granularity']=='PASSAGE' for r in group)))
result=dict(W11_pilot=a,W12_VAL=b,selected_export_coverage=coverage,limitations='Detailed full C/D prepared selections were removed before this audit. Selected-unit results cover only retained frozen attribution exports, not all 1005 VAL queries. Whole passages are not invalid sentence units.',QASPER_native_sentences=False,HotpotQA_native_sentences=True,SciFact_native_sentences=True)
(out/'qasper_frozen_splitter_impact.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
lines=['# Read-only historical QASPER splitter impact audit','',
'QASPER used the frozen fallback splitter, not native sentence offsets. Week-11 reconstructs source papers from the official TRAIN Arrow file using the saved pilot paper indices; Week-12 uses all frozen VAL papers from official validation. Both source constructors remove `BIBREF` identifiers before segmentation. No answer was regenerated and no splitter/result was modified.','',
'## Exact source-population census','', '| Source population | Papers | Passages | Frozen units | Citation-aware counterfactual segments | Affected frozen units | Affected % |','|---|---:|---:|---:|---:|---:|---:|']
for name,x in [('W11 granularity pilot',a),('W12 frozen VAL',b)]:
 c=x['counts'];lines.append('| '+name+' | '+' | '.join(str(c[k]) for k in ('papers','passages','frozen_units','counterfactual_segments','affected_units'))+f" | {c['affected_percent']:.4f}% |")
lines+=['','Affected means an existing frozen unit subdivides under the explicit citation/Markdown LiveDemo adapter. This is a counterfactual boundary diagnostic, not proof of changed answer quality. Paragraph/chunk-edge effects, abbreviations and ordinary NLP ambiguity remain. All passages count once per unique paper, not once per query.','',
'## Saved FIXED_SENTENCE/QBGF context coverage','',
'| Method | Budget | Available queries | Selected units | Sentence occurrences | Resolved | Affected | Unresolved | Whole passages |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for c in coverage:lines.append('| '+' | '.join(str(c[k]) for k in ('method','budget','available_queries','selected_unit_occurrences','sentence_unit_occurrences','resolved_sentence_occurrences','affected_sentence_occurrences','unresolved_sentence_occurrences','whole_passage_occurrences'))+' |')
lines+=['','The full C/D prepared/answer shards are no longer present. The retained attribution export is a subset, so it cannot certify every selected unit across the 1005-query QASPER validation population. Unresolved IDs, if any, remain unknown rather than unaffected. Complete-passage QBGF selections naturally contain multiple sentences and are not counted as faulty sentence units.','',
'## Native boundaries in the other datasets','',
'`src/week11/datasets.py` builds HotpotQA offsets from the provided sentence arrays and SciFact offsets from each abstract sentence array; document titles are also explicit native units. `src/week11/hierarchy.py` prioritizes these native offsets. The fallback citation-regex mechanism therefore does not define these native sentence boundaries. This does not certify that every dataset-native unit is a perfect grammatical sentence.','',
'## Interpretation and limitation','',
'Frozen research results retain their original preprocessing exactly. Any nonzero impact is a documented limitation of the fallback unit definition, not permission to repair historical answers or infer a new method-quality improvement. The impact counts do not measure downstream score sensitivity. If observed selected-unit coverage is small, do not generalize it to absent historical contexts.','',
'Full affected texts, offsets, unit IDs and counterfactual segments are saved in the adjacent JSON. The audit accessed source text and saved identity metadata only; it did not access QASPER TEST or run inference.']
(out/'qasper_frozen_splitter_impact.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:v['counts'] for k,v in [('W11',a),('W12',b)]},indent=2),flush=True)

"""Validated result extraction and deterministic example curation; no model work."""
import json
from functools import lru_cache
from pathlib import Path
from collections import Counter
from demo.constants import ROOT, DATASETS, BUDGETS, METHODS
from demo.data_loader import load, summaries, DataError

def findings():
    val=summaries('validation');conf=summaries('confirmation')
    counts=Counter(r['category'] for r in load('units'))
    comparisons=load('consistency')
    return dict(validation=val,confirmation=conf,contrasts=load('contrasts'),primary=load('primary'),
                fusion=load('fusion'),correlations=load('correlations'),unit_counts=dict(counts),
                directions=comparisons)

def lookup(rows,dataset,method,budget):
    found=[r for r in rows if (r['dataset'],r['method'],r['budget'])==(dataset,method,budget)]
    if len(found)!=1:raise DataError('Condition missing or duplicated')
    return found[0]

def profiles():
    rs=load('profiles');validate_profile_keys(rs)
    return rs

def validate_profile_keys(rs):
    keys=set()
    for r in rs:
        key=(r['dataset'],r['query_id'],r['method'],r['budget'])
        if key in keys or r['dataset'] not in DATASETS or r['budget'] not in BUDGETS:raise DataError('Invalid profile identity')
        keys.add(key)

def curated_ids(dataset, limit=12):
    """Prefer retained real text examples; add strongest observed paired profile differences.
    This is explicitly outcome-curated VAL storytelling, not a representative sample.
    """
    chosen=[]
    for r in load('qualitative'):
        if r['dataset']==dataset and r['query_id'] not in chosen:chosen.append(r['query_id'])
    grouped={}
    for r in profiles():
        if r['dataset']==dataset:grouped.setdefault(r['query_id'],[]).append(r)
    ordered=sorted(grouped,key=lambda q:(-max(r['full_score'] for r in grouped[q])+min(r['full_score'] for r in grouped[q]),q))
    chosen.extend(q for q in ordered if q not in chosen)
    return chosen[:limit]

@lru_cache(maxsize=3)
def _index_file(path,mtime):return json.loads(Path(path).read_text())

def replay_index():
    p=ROOT/'demo/cache/replay_index.json'
    if not p.exists():return {}
    obj=_index_file(str(p),p.stat().st_mtime_ns)
    # A recovered index is only reusable for the exact primary exports it references.
    from demo.data_loader import provenance
    for name,identity in obj['sources'].items():
        if provenance(name)['sha256']!=identity['sha256']:raise DataError('Recovered replay index is stale: '+name)
    return obj['queries']

def get_query(dataset,query_id,index=None):
    if dataset not in DATASETS:raise DataError('Unknown dataset')
    matching=[r for r in profiles() if r['dataset']==dataset and r['query_id']==query_id]
    if not matching:raise DataError('Query ID not present in saved VAL attribution cohort')
    item=(index if index is not None else replay_index()).get(dataset+':'+query_id,{})
    qualitative=[r for r in load('qualitative') if r['dataset']==dataset and r['query_id']==query_id]
    return dict(dataset=dataset,query_id=query_id,profiles=matching,question=item.get('question') or (qualitative[0]['question'] if qualitative else None),
                gold=item.get('gold') or (qualitative[0]['gold'] if qualitative else None),hierarchy=item.get('hierarchy',[]),
                source=item.get('source'),qualitative=qualitative,available=bool(item))

def method_replay(query,method,budget):
    rs=[r for r in query['profiles'] if r['method']==method and r['budget']==budget]
    if not rs:return None
    profile=rs[0];units=[dict(r) for r in load('units') if r['dataset']==query['dataset'] and r['query_id']==query['query_id'] and r['method']==method and r['budget']==budget]
    byid={u['unit_id']:u for u in query['hierarchy']}
    saved={u['unit_id']:u['text'] for q in query['qualitative'] if q['method']==method and q['budget']==budget for u in q['selected_evidence']}
    for u in units:
        unit=byid.get(u['unit_id']);u.update(source=unit,text=unit['source_text'] if unit else saved.get(u['unit_id']))
    texts=[r['original_answer'] for r in query['qualitative'] if r['method']==method and r['budget']==budget]
    if texts:answer=texts[0];answer_origin='Saved original answer'
    else:
        unchanged={r['ablated_answer'] for r in units if r.get('exact_flip') is False}
        if len(unchanged)>1:raise DataError('Unchanged-ablation answers disagree')
        answer=next(iter(unchanged),None);answer_origin='Recovered from saved exact-unchanged ablation output' if answer is not None else 'Original answer text unavailable'
    return dict(profile=profile,units=sorted(units,key=lambda u:u['unit_index']),answer=answer,answer_origin=answer_origin)

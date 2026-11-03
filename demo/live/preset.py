"""Explicitly approved static diagnostic, never a tuning set or correctness oracle."""
import json
from functools import lru_cache
from pathlib import Path
@lru_cache(maxsize=1)
def load_preset():return json.loads((Path(__file__).resolve().parents[1]/'assets/presets/world_cup.json').read_text())
def is_preset(query,passages):
    p=load_preset();return query.strip()==p['query'] and list(passages)==p['passages']
def passage_status(result):
    rows=[]
    for pid,topic in [('P7','Tournament workload'),('P9','Abandoned three-team format')]:
        ranked=next((x for x in result['ranking'] if x['unit']['document_id']==pid),None)
        candidate=ranked is not None and ranked['in_pool']
        chosen=[u for u in result['selection']['units'] if u['document_id']==pid]
        all_fine=[u for u in result['sentences'] if u['document_id']==pid]
        full=any(u['unit_type']=='PASSAGE' for u in chosen)
        allocation='NOT IN CANDIDATE POOL' if not candidate else 'DROPPED' if not chosen else 'SELECTED · FULL PASSAGE' if full else 'SELECTED · ALL SOURCE UNITS' if all_fine and len(chosen)==len(all_fine) else 'SELECTED · PARTIAL REPRESENTATION'
        option=next((o['label'] for o in result.get('qbgf_options',[]) if o['passage_id']==pid and o['selected']),None)
        rows.append(dict(Passage=pid,Topic=topic,Retrieval='RETRIEVED' if candidate else 'NOT RETRIEVED',Rank=ranked['rank'] if ranked else None,Allocation=allocation,Selected_units=len(chosen),QBGF_option=option or '—'))
    return rows

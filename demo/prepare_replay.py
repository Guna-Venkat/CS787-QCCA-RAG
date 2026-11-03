"""CPU-only recovery of selected units from raw VAL sources, via canonical code.
Never recomputes retrieval, similarity, allocation decisions, predictions or scores.
Writes a small derived index ONLY under demo/cache/.
Run with sara_env/bin/python -m demo.prepare_replay.
"""
import json,sys,time
from pathlib import Path
from demo.constants import ROOT
from demo.data_loader import load, provenance, sha
from demo.result_registry import curated_ids

def main(limit=12):
    from transformers import AutoTokenizer
    from src.week11.hierarchy import build
    from src.week11.datasets import hotpot, scifact, read_jsonl
    from src.week12.data import native_qasper
    from src.week12.algorithm import serialize
    cfg=json.loads((ROOT/'results/week12/qbgf/v3/protocol/protocol_config.json').read_text())
    spec=cfg['models']['Qwen']
    tok=AutoTokenizer.from_pretrained(spec['id'],revision=spec['revision'],local_files_only=True)
    freeze=json.loads((ROOT/'results/week12/qbgf/v3/protocol/W12_SPLIT_FREEZE.json').read_text())
    wanted={d:set(curated_ids(d,limit)) for d in ('HotpotQA','QASPER','SciFact')}
    for d,ids in wanted.items():
        allowed={r['query_id'] for r in freeze['datasets'][d]['VAL']}
        if not ids<=allowed:raise ValueError('Replay includes non-VAL IDs')
    raw=Path(cfg['datasets']['QASPER']['raw_path']).with_name('qasper-validation.arrow')
    hp=ROOT/cfg['datasets']['HotpotQA']['path'];sp=ROOT/cfg['datasets']['SciFact']['path'];cp=ROOT/cfg['datasets']['SciFact']['corpus_path']
    xs=hotpot([r for r in json.loads(hp.read_text()) if r['_id'] in wanted['HotpotQA']])
    xs+=native_qasper(raw,allowed_ids=wanted['QASPER'])
    # Full source corpus is necessary to match stored selected IDs, never gold-only retrieval.
    xs+=scifact([r for r in read_jsonl(sp) if str(r['id']) in wanted['SciFact']],read_jsonl(cp))
    allrows=load('units');doc_cache={};out={};started=time.monotonic()
    for n,e in enumerate(xs):
        key=e['dataset']+':'+e['query_id']
        measured=[r for r in allrows if r['dataset']==e['dataset'] and r['query_id']==e['query_id']]
        selected_ids={r['unit_id'] for r in measured};found={};all_units={}
        for doc_index,doc in enumerate(e['inference']['documents']):
            cachekey=(doc['dataset'],doc['document_id'])
            if cachekey not in doc_cache:doc_cache[cachekey]=build([doc],tok,cfg['hierarchy'])
            units=[dict(u,document_order=[doc_index]+u['document_order'][1:]) for u in doc_cache[cachekey]]
            if not any(u['unit_id'] in selected_ids for u in units):continue
            # Keep the whole parent hierarchy so passage/sentence decomposition is source-exact.
            parents={u['passage_id'] for u in units if u['unit_id'] in selected_ids}
            for u in units:
                if u['passage_id'] in parents:all_units[u['unit_id']]=dict(u,serialized_block_tokens=serialize([u],tok)['evidence_tokens'])
                if u['unit_id'] in selected_ids:found[u['unit_id']]=u
        if set(found)!=selected_ids:raise ValueError(f'{key}: {len(selected_ids-set(found))} selected unit IDs could not be recovered exactly')
        for r in measured:
            if found[r['unit_id']]['token_count']!=r['tokens']:raise ValueError('Recovered token count mismatch')
        # Verify selected evidence totals using exact original serialization.
        profiles=load('profiles')
        for p in profiles:
            if p['dataset']==e['dataset'] and p['query_id']==e['query_id']:
                selected=sorted([r for r in measured if r['method']==p['method'] and r['budget']==p['budget']],key=lambda r:r['unit_index'])
                serial=serialize([found[r['unit_id']] for r in selected],tok)
                if serial['evidence_tokens']!=p['evidence_tokens']:raise ValueError('Saved evidence cost does not reproduce')
        out[key]=dict(question=e['question'],gold={k:v for k,v in e['evaluation'].items() if k!='gold_spans'},hierarchy=list(all_units.values()),
                      source='Canonical source hierarchy recovered by exact unit ID + source substring + token cost. Historical selection membership comes from the saved VAL LOO export; no retrieval or allocation rerun.')
        print(f'CPU replay recovery {n+1}/{len(xs)}: {key}; {len(found)} exact selected units; {time.monotonic()-started:.1f}s',flush=True)
    dest=ROOT/'demo/cache';dest.mkdir(parents=True,exist_ok=True)
    metadata=dict(sources={k:provenance(k) for k in ['units','profiles','qualitative']},queries=out,
                  source_identities={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):sha(p) for p in [hp,sp,cp,raw,ROOT/'src/week11/hierarchy.py',ROOT/'src/week11/propositions.py',ROOT/'src/week12/algorithm.py']},tokenizer=spec,
                  limitations=['Original complete top10 ranks/scores and QBGF choices removed with detailed caches. Missing values remain unavailable.','Source hierarchy is verified recovery; unselected hierarchy units are not newly evaluated representations.'])
    (dest/'replay_index.json').write_text(json.dumps(metadata,separators=(',',':')))
    print('Saved',len(out),'VAL queries to demo/cache/replay_index.json',flush=True)
if __name__=='__main__':main()

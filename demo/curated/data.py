"""VAL-only source loading. Never reads confirmation or historical QASPER TEST."""
from pathlib import Path
from .protocol import ROOT,DATASETS

def examples(c,m):
    import json
    from src.week11.datasets import hotpot,scifact,read_jsonl
    from src.week12.data import native_qasper
    wanted={d:set(m['queries'][d]) for d in DATASETS}
    hp=ROOT/c['datasets']['HotpotQA']['path'];sp=ROOT/c['datasets']['SciFact']['path'];cp=ROOT/c['datasets']['SciFact']['corpus_path']
    xs=hotpot([r for r in json.loads(hp.read_text()) if r['_id'] in wanted['HotpotQA']])
    xs+=scifact([r for r in read_jsonl(sp) if str(r['id']) in wanted['SciFact']],read_jsonl(cp))
    xs+=native_qasper(Path(c['datasets']['QASPER']['raw_path']).with_name('qasper-validation.arrow'),allowed_ids=wanted['QASPER'])
    out={(e['dataset'],e['query_id']):e for e in xs}
    if set(out)!={(d,q) for d in DATASETS for q in wanted[d]}:raise ValueError('Curated example source IDs disagree')
    return out

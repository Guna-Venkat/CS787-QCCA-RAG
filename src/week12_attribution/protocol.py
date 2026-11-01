import json,copy,sqlite3,collections,random
from pathlib import Path
import numpy as np
from src.week12.protocol import ROOT,root,digest,sha,identity,Store,immutable,cohort,guard
BUDGETS=[256,1024]
CANDIDATES=['FIXED_SENTENCE','C2','H3','RAW_SKIP','BM25_PLAIN_RAG','QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY']

def directory(cfg,smoke=False):return ROOT/'results/week12/attribution'/(('smoke/'+study_identity(cfg,True)['code'][:12]) if smoke else 'v1')
def study_identity(cfg,smoke=False):
    return dict(parent=identity(cfg),code=digest(dict({p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},runner=sha(ROOT/'scripts/week12/run_attribution.py'))),study='posthoc-input-ablation-v1',seed=42,epsilon=0,budgets=BUDGETS,synthetic=smoke,counts=1 if smoke else 50,confirmation_access=False)
def store(path):return Store(path,json.loads((path/'identity.json').read_text()))
_SOURCES={}
def source(cfg,phase,kind):
    key=(str(root(cfg)),phase,kind)
    if key not in _SOURCES:
        s=store(root(cfg)/('c_extension/v1' if phase=='C-EXT' else 'validation/'+phase)/kind)
        meta=s.identity.get('parent',s.identity)
        if meta['code_hash']!=identity(cfg)['code_hash'] or meta['config_hash']!=digest(cfg):raise ValueError('Frozen source identity differs')
        _SOURCES[key]=s
    return _SOURCES[key]
def locate(cfg,method):return 'C-EXT' if method=='BM25_PLAIN_RAG' else 'D' if method in ('QBGF_SENTENCE_ONLY','QBGF_PASSAGE_ONLY') else 'C'
def original(cfg,d,q,method,budget,kind='rows'):
    key=[d,q,method,budget]+(['BASE_FP16'] if kind=='rows' else [])
    row=source(cfg,locate(cfg,method),kind).get(key)
    if row is None:raise ValueError('Required original result/selection missing: '+str(key))
    return row

def stratified_sample(rows,n=50,seed=42):
    """Proportional strata, seeded sorted IDs; prefer distinct groups (QASPER papers)."""
    if len(rows)<n:raise ValueError('Insufficient frozen VAL examples')
    pools=collections.defaultdict(list)
    for r in rows:pools[r['stratum']].append(r)
    exact={s:n*len(v)/len(rows) for s,v in pools.items()};quota={s:int(v) for s,v in exact.items()}
    for s in sorted(pools,key=lambda s:(-(exact[s]-quota[s]),s))[:n-sum(quota.values())]:quota[s]+=1
    rng=random.Random(seed);out=[];groups=set()
    for s in sorted(pools):
        candidates=sorted(pools[s],key=lambda r:r['query_id']);rng.shuffle(candidates)
        candidates=sorted(candidates,key=lambda r:r['group'] in groups)
        for r in candidates:
            if sum(x['stratum']==s for x in out)>=quota[s]:break
            if r['dataset']=='QASPER' and r['group'] in groups:continue
            out.append(r);groups.add(r['group'])
        # Explicit fallback only if a stratum cannot satisfy its quota using unique papers.
        for r in candidates:
            if sum(x['stratum']==s for x in out)>=quota[s]:break
            if r not in out:out.append(r);groups.add(r['group'])
    assert len(out)==n and len({r['query_id'] for r in out})==n
    return sorted(out,key=lambda r:r['query_id'])

def freeze(cfg):
    guard();f=directory(cfg);p=ROOT/'results/week12/attribution/ATTRIBUTION_QUERY_FREEZE.json'
    if p.exists():
        obj=json.loads(p.read_text())
        if obj['identity']!=study_identity(cfg):raise ValueError('Study identity changed; preserve frozen study and use a new version')
        validate_membership(cfg,obj);return obj
    for phase in ('C','D'):
        status=json.loads((root(cfg)/'validation'/phase/'status.json').read_text())
        if not status.get('complete') or not status.get('full_generation'):raise ValueError('Completed full C/D required')
    from src.week12.statistics import metric
    from src.week12.splits import read_freeze
    split=read_freeze(cfg);selected={};comparators={};results=[]
    for d,records in cohort(cfg).items():
        metadata=[]
        for item in records:
            r=original(cfg,d,item['query_id'],'NONE',256)
            # Sampling uses ONLY membership/stratum metadata, never outcomes or method differences.
            metadata.append(dict(dataset=d,query_id=item['query_id'],group=r['cluster_id'],paper_id=r['paper_id'],stratum=str(r['gold']) if d=='SciFact' else str(r.get('answer_type','unknown'))))
        selected[d]=stratified_sample(metadata)
        for b in BUDGETS:
            candidates=[]
            for m in CANDIDATES:
                if m=='C2' and b!=256:continue
                rs=[original(cfg,d,x['query_id'],m,b) for x in records]
                candidates.append(dict(method=m,score=metric(rs,cfg['statistics']['primary_metric'][d]),mean_prompt_tokens=float(np.mean([r['prompt_tokens'] for r in rs])),n_queries=len(rs)))
            chosen=min(candidates,key=lambda r:(-r['score'],r['mean_prompt_tokens'],r['method']))
            comparators[d+':'+str(b)]=chosen['method'];results.append(dict(dataset=d,budget=b,chosen=chosen['method'],candidates=candidates))
    obj=dict(identity=study_identity(cfg),split_hash=split['split_hash'],selected=selected,comparators=comparators,comparator_results=results,sampling_rule='50 per dataset; proportional answer-type/class strata; seed42 sorted-ID shuffle; prefer unique QASPER papers; never filter by outcomes',comparator_rule='Maximum complete full-VAL task metric among fixed-sentence/C2/H3/SFR RAW_SKIP/BM25 RAW_SKIP and D single-granularity controls; exact ties lower mean prompt tokens, then method name. Excludes multi-granular NO_QUERY and specialists.',epsilon=0,budgets=BUDGETS,methods_per_cell=2,unit_definition='Selected sentence/PROPOSITION span individually; FULL/RAW_SKIP passage one unit; no resplitting FULL',inference='Exact frozen BASE_FP16 Qwen reader/parser/tokenizer/decoding. Only selected evidence differs.',statistics='10000 paper/query/document-group bootstrap resamples, seed42; post-hoc diagnostic, no primary policy tuning',retention='Only define normalized gain retention when full score minus NONE score >0',causal_ties='Original source order then stable unit ID; never break causal ties using relevance',optional_attention=False,optional_replacement=False)
    validate_membership(cfg,obj);immutable(p,obj);return obj

def validate_membership(cfg,obj):
    from src.week12.splits import read_freeze
    split=read_freeze(cfg)
    for d,rows in obj['selected'].items():
        selected={r['query_id'] for r in rows};val={r['query_id'] for r in split['datasets'][d]['VAL']};confirm={r['query_id'] for r in split['datasets'][d]['CONFIRM']}
        if len(rows)!=50 or len(selected)!=50 or not selected<=val or selected&confirm:raise ValueError('Attribution split/count/confirmation guard failed')
    return True

def cached_vectors(cfg,texts,role='document'):
    """Exact canonical embedding keys; read-only SQLite; never instantiate an encoder/model."""
    from src.week10.root_cause.encoding import Encoder
    obj=Encoder.__new__(Encoder);obj.spec=cfg['models']['SFR']
    p=ROOT/cfg['cache_root']/'embeddings/embeddings.sqlite3';db=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True);vs=[]
    try:
        for text in texts:
            r=db.execute('SELECT vector,dim FROM vectors WHERE key=?',(obj.key(text,role),)).fetchone()
            if r is None:raise ValueError('Frozen embedding not cached; no silent recomputation')
            v=np.frombuffer(r[0],dtype=np.float32).copy()
            if len(v)!=r[1] or not np.isfinite(v).all():raise ValueError('Invalid cached vector')
            vs.append(v)
    finally:db.close()
    return np.asarray(vs)

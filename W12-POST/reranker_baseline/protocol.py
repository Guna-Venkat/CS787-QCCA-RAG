"""All new files are scoped to W12-POST; historical stores are read-only."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'W12-POST'
OUT=HOME/'results/minilm_reranker'
LABEL='POST_W12_RERANKER_BASELINE_EXPLORATORY'
METHOD='MINILM_CE_SENTENCE'
MODEL='cross-encoder/ms-marco-MiniLM-L6-v2'
REVISION='233902d25c440f23af6f7d6e94d2946bac0bee0a'
DATASETS=['HotpotQA','QASPER','SciFact']
BUDGETS=[256,1024]
CONTROLS=['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF_V1','BEA_V2']
METHODS=CONTROLS+[METHOD]
CONFIG=dict(label=LABEL,method=METHOD,model=MODEL,revision=REVISION,max_length=512,
    activation='raw_logit_identity',device='cpu',dtype='float32',batch_size=32,
    ranking='score DESC, passage rank ASC, source position ASC, stable ID ASC',
    allocation='global ranking; include if exact canonical serialized cost fits; skip otherwise; exhaust candidates; present in source order',
    evidence_budgets=BUDGETS,retriever='SFR',retrieval_width=10,queries_per_dataset=50,
    reader='BASE_FP16',lora=False,resume=True,overwrite=False,resamples=10000,seed=42,
    exploratory_holm_family=24,automatic_expansion=False)
NOTICE='POST-W12 exploratory matched-cohort comparison on previously analyzed development data. No fresh holdout, confirmation or test claim. Official Week-12 policy unchanged.'

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())

def save(path,obj,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(HOME.resolve()):raise ValueError('New writes must remain inside W12-POST')
    content=obj if isinstance(obj,str) else json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and not replace:
        if path.read_text()!=content:raise ValueError('Incompatible existing reranker artifact; overwrite prohibited: '+str(path))
        return
    fd,name=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'w') as stream:stream.write(content);stream.flush();os.fsync(stream.fileno())
        if replace:os.replace(name,path)
        else:
            try:os.link(name,path)
            except FileExistsError:
                if path.read_text()!=content:raise ValueError('Concurrent incompatible reranker artifact')
    finally:
        if os.path.exists(name):os.unlink(name)

class Store:
    def __init__(self,path,identity):self.path=Path(path);self.identity=identity
    def file(self,key):return self.path/(digest(key)+'.json')
    def get(self,key):
        p=self.file(key)
        if not p.exists():return None
        v=read(p)
        if (v['key']!=key or v['identity_hash']!=digest(self.identity)
            or v['checksum']!=digest(v['row']) or v['label']!=LABEL):
            raise ValueError('Reranker cache identity/checksum mismatch: '+str(p))
        return v['row']
    def put(self,key,row):
        save(self.file(key),dict(label=LABEL,key=key,identity_hash=digest(self.identity),checksum=digest(row),row=row))
        return row

@contextlib.contextmanager
def lock():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.lock').open('a') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another reranker stage is active')
        yield

def versions():
    return {k:importlib.metadata.version(k) for k in ['torch','transformers','sentence-transformers','huggingface_hub','numpy']}

def context(persist=False):
    from src.post_w12.bea_v2_posthoc.protocol import context as original_context, Store as SourceStore, keys
    c,old,parent,v1,splits,cohort,f=original_context()
    path=ROOT/c['output_root'];complete=read(path/'COMPLETE.json')
    expected=keys(cohort)
    if (complete['logical_rows']!=1200 or complete['identity_hash']!=f['identity_hash']
        or complete['keys_hash']!=digest(expected)):
        raise ValueError('Required exact matched-cohort BEA_V2 comparison is incomplete or incompatible')
    source=SourceStore(path/'rows',f['identity'])
    if {p.name for p in source.path.glob('*.json')}!={source.file(k).name for k in expected}:
        raise ValueError('Stored controls do not share the exact expected membership')
    source_hashes={}
    for d in DATASETS:
        members=cohort['selected'][d]
        if len(members)!=50 or len({m['query_id'] for m in members})!=50:
            raise ValueError('Exactly 50 frozen examples/dataset are required')
        for m in CONTROLS:
            for b in BUDGETS:
                got=set()
                for member in members:
                    q=member['query_id'];key=[d,q,m,b];r=source.get(key)
                    if (r is None or [r['dataset'],r['query_id'],r['method'],r['budget']]!=key
                        or r['cluster_id']!=member['group'] or r['reader']!='BASE_FP16'
                        or r['model_hash']!=digest([parent['models']['Qwen'],'BASE_FP16',None])):
                        raise ValueError('Baseline method/query/budget/group/model mismatch: '+str(key))
                    got.add(q);source_hashes[str(source.file(key).relative_to(ROOT))]=sha(source.file(key))
                if got!={x['query_id'] for x in members}:raise ValueError('Method membership mismatch')
    paths=list(Path(__file__).parent.glob('*.py'))+[HOME/'run_minilm_reranker.py']
    implementation={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    identity=dict(label=LABEL,config=CONFIG,implementation=implementation,packages=versions(),
        source_identity_hash=f['identity_hash'],source_complete_sha=sha(path/'COMPLETE.json'),
        cohort=cohort['selected'],baseline_row_hashes=source_hashes,
        source_protected_inputs=c['protected_inputs'],
        reader=parent['models']['Qwen'],prompt_sha=v1['identity']['prompt_sha'],parser_sha=v1['identity']['parser_sha'],
        serializer=v1['identity']['serializer'],decoding=v1['identity']['decoding'],retrieval=v1['input_identity'])
    record=dict(label=LABEL,identity=identity,identity_hash=digest(identity),notice=NOTICE)
    p=OUT/'protocol/IDENTITY.json'
    if persist:save(p,record)
    elif p.exists() and read(p)!=record:raise ValueError('Reranker implementation/inputs changed after protocol freeze')
    return dict(old=old,parent=parent,v1=v1,splits=splits,cohort=cohort,source=source,record=record)

def stores(ctx,mode='full'):
    if mode not in ['full','smoke']:raise ValueError('Unknown execution mode')
    ident=ctx['record']['identity']
    return (Store(OUT/'cache/scores',ident),Store(OUT/(mode+'/selections'),ident),
            Store(OUT/('rows' if mode=='full' else 'smoke/reader_rows'),ident),Store(OUT/'cache/answers',ident))

def members(ctx,mode):
    if mode=='full':return {d:list(ctx['cohort']['selected'][d]) for d in DATASETS}
    if mode=='smoke':return {d:sorted(ctx['cohort']['selected'][d],key=lambda r:r['query_id'])[:1] for d in DATASETS}
    raise ValueError('Unknown mode')

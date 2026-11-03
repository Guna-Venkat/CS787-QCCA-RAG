import functools,hashlib,json,os,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'demo';OUT=APP/'replay_data/v1'
DATASETS=('QASPER','HotpotQA','SciFact');RETRIEVERS=('SFR','BM25','MiniLM');BUDGETS=(256,1024)
METHODS=('NONE','RAW_SKIP','FIXED_SENTENCE','C2','H3','QBGF_FULL','PROPOSITION','MULTI_SENTENCE_2','MULTI_SENTENCE_4')
LABELS={'NONE':'No evidence','RAW_SKIP':'Whole passages','FIXED_SENTENCE':'Query-selected sentences','C2':'Top-eight sentences','H3':'Fixed hierarchy','QBGF_FULL':'Adaptive QBGF','PROPOSITION':'Exact-source propositions','MULTI_SENTENCE_2':'Two sentences per passage','MULTI_SENTENCE_4':'Four sentences per passage'}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def immutable(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);text=json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n'
    if p.exists():
        if p.read_text()!=text:raise ValueError('Incompatible immutable demo artifact: '+str(p))
        return x
    fd,tmp=tempfile.mkstemp(dir=p.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        try:os.link(tmp,p)
        except FileExistsError:
            if p.read_text()!=text:raise ValueError('Concurrent demo artifact mismatch')
    finally:os.unlink(tmp)
    return x
class Store:
    def __init__(self,p,identity):self.path=Path(p);self.identity=identity;immutable(self.path/'identity.json',identity)
    def get(self,key):
        p=self.path/(digest(key)+'.json')
        if not p.exists():return None
        x=read(p)
        if x['key']!=key or x['identity']!=digest(self.identity) or digest(x['value'])!=x['sha256']:raise ValueError('Demo shard checksum/identity mismatch')
        return x['value']
    def put(self,key,value):return immutable(self.path/(digest(key)+'.json'),dict(key=key,value=value,sha256=digest(value),identity=digest(self.identity)))

def config():
    from demo.constants import FREEZE_SHA
    from demo.data_loader import freeze_check
    freeze_check()
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='gagan':raise ValueError('Use gagan')
    c=read(ROOT/'results/week12/qbgf/v3/protocol/protocol_config.json')
    models=read(ROOT/'results/week10/root_cause/v1/protocol.json')['models']
    c['models']['MiniLM']=models['MiniLM']
    # Canonical Lab's only writable path is redirected inside the app, even for fixtures.
    c['cache_root']='demo/replay_data/v1/cache';c['output_root']='demo/replay_data/v1'
    return c

def manifest(c):
    from demo.constants import FREEZE_SHA
    from demo.data_loader import freeze_check
    freeze_check()
    split=ROOT/'results/week12/qbgf/v3/protocol/W12_SPLIT_FREEZE.json'
    source=read(APP/'example_catalog.json')['queries'];f=read(split)
    chosen={d:source[d][:10] for d in DATASETS}
    for d in DATASETS:
        allowed={str(r['query_id']) for r in f['datasets'][d]['VAL']}
        if len(chosen[d])!=10 or len(set(chosen[d]))!=10 or not set(chosen[d])<=allowed:raise ValueError('Ten unique frozen VAL examples per dataset required')
    dependencies=[ROOT/'src/week12/algorithm.py',ROOT/'src/week12/reader.py',ROOT/'src/week12/laboratory.py',ROOT/'src/week10/root_cause/encoding.py',ROOT/'src/week11/hierarchy.py',ROOT/'src/week11/propositions.py',ROOT/'src/week11/evaluation.py',ROOT/'src/week11/datasets.py',ROOT/'src/week12/data.py',ROOT/'src/week6/evaluation.py',ROOT/'src/week9/retrieval.py']+sorted((APP/'curated').glob('*.py'))
    sources={str(p):sha(p) for p in [ROOT/c['datasets']['HotpotQA']['path'],ROOT/c['datasets']['SciFact']['path'],ROOT/c['datasets']['SciFact']['corpus_path'],Path(c['datasets']['QASPER']['raw_path']).with_name('qasper-validation.arrow')]}
    obj=dict(version='curated-demo-v1',scope='new outcome-curated VAL illustrations; not benchmark results or historical replay',queries=chosen,
             retrievers=list(RETRIEVERS),methods=list(METHODS),budgets=list(BUDGETS),reader='Qwen BASE_FP16; no LoRA',models=c['models'],config_hash=digest(c),config=c,
             freeze_sha256=FREEZE_SHA,split_sha256=sha(split),source_hashes=sources,code_hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies},expected_rows=30*3*9*2)
    return immutable(OUT/'manifest.json',obj)

def grid(m):return [(d,q,r,b,a) for d in DATASETS for q in m['queries'][d] for r in RETRIEVERS for b in BUDGETS for a in METHODS]
def store(m,name):return Store(OUT/name,dict(manifest_hash=digest(m),kind=name))

"""Read-only frozen dependencies, isolated immutable outputs, metadata-only cohorts."""
import gzip,json,os,tempfile,subprocess,functools,fcntl
from pathlib import Path
from src.week12.protocol import ROOT,digest,sha

LABEL='POST_W12_EXPLORATORY'
FREEZE_SHA='d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b'
SPLIT_HASH='78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117'
ATTRIBUTION_SHA='fe529e21126d36d001083859d92ef13196523bc225318aff2bb47b26f7a05218'
DATASETS=('HotpotQA','QASPER','SciFact')
METHODS=['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF']

def read(p):return json.loads(Path(p).read_text())
def load(path=None):
    c=read(path or ROOT/'W12-POST/config.json')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='gagan':raise ValueError('Use branch gagan')
    if c['label']!=LABEL or c['methods']!=METHODS or c['budgets']!=[256,1024]:raise ValueError('Exploratory registry differs')
    if not output(c).is_relative_to(ROOT/'results/post_w12/bea_qbgf'):raise ValueError('New outputs must stay isolated')
    if c['cache']['overwrite_existing'] or c['cache']['force_regenerate']:
        raise ValueError('For deliberate regeneration choose a fresh output version; never overwrite original results')
    return c
def output(c):return (ROOT/c['output_root']).resolve()

def exclusive(function):
    @functools.wraps(function)
    def wrapped(c=None,*args,**kwargs):
        c=c or load();directory=output(c);directory.mkdir(parents=True,exist_ok=True)
        with (directory/'.execution.lock').open('a') as handle:
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another BEA stage is active; do not start duplicate runs')
            return function(c,*args,**kwargs)
    return wrapped

def save(path,obj,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'results/post_w12/bea_qbgf'):raise ValueError('Attempt to write outside exploratory outputs')
    path.parent.mkdir(parents=True,exist_ok=True)
    content=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists() and not replace:
        if path.read_text()!=content:raise ValueError('Incompatible existing artifact: '+str(path)+'; use a new output version')
        return obj
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'w') as f:f.write(content);f.flush();os.fsync(f.fileno())
        if replace:os.replace(tmp,path)
        else:
            try:os.link(tmp,path)
            except FileExistsError:
                if path.read_text()!=content:raise ValueError('Concurrent incompatible result')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return obj

def text(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'results/post_w12/bea_qbgf'):raise ValueError('Outside exploratory outputs')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(value);os.replace(tmp,path)

def frozen():
    a=ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json';b=ROOT/'results/week12/qbgf/v3/W13_FINAL_METHOD_FREEZE.json'
    if sha(a)!=FREEZE_SHA or a.read_bytes()!=b.read_bytes():raise ValueError('Original final freeze changed')
    f=read(a)
    if f['method']!='FIXED_SENTENCE' or f['retain_lora'] or f['reader']!='BASE_FP16':raise ValueError('Frozen main policy differs')
    return f

def split(parent):
    p=ROOT/parent['output_root']/'protocol/W12_SPLIT_FREEZE.json'
    # Never call the historical auto-decompressor: it writes into frozen directories.
    s=json.loads(p.read_text() if p.exists() else gzip.decompress(p.with_suffix('.json.gz').read_bytes()))
    if s['split_hash']!=SPLIT_HASH or digest({k:v for k,v in s.items() if k!='split_hash'})!=SPLIT_HASH:
        raise ValueError('Frozen split identity mismatch')
    if s['config_hash']!=digest(parent):raise ValueError('Original config does not match the frozen split')
    return s

def verify(c,verify_data=True):
    from src.week12.protocol import load as parent_load,code_hash
    f=frozen();parent=parent_load(ROOT/c['parent_config']);s=split(parent)
    if digest(parent)!=f['identity']['config_hash'] or code_hash()!=f['identity']['code_hash']:
        raise ValueError('Frozen scientific code/config changed; no silent reinterpretation')
    if sha(ROOT/'src/week12/reader.py')!=f['prompt_source_hash'] or f['parser_hash']!=f['prompt_source_hash']:
        raise ValueError('Frozen prompt/parser changed')
    if verify_data:
        for p,h in s['source_hashes'].items():
            if sha(p)!=h:raise ValueError('Frozen source changed: '+p)
    import importlib.metadata
    versions={k:importlib.metadata.version(k) for k in f['identity']['packages']}
    if versions!=f['identity']['packages']:raise ValueError('Frozen inference package versions differ')
    if parent['models']!=f['identity']['models'] or parent['generation']['precision']!='float16':raise ValueError('Reader/retriever/precision changed')
    return parent,f,s

def implementation():
    paths=list(Path(__file__).parent.glob('*.py'))+list((ROOT/'W12-POST/tests').glob('*.py'))+[ROOT/'W12-POST/config.json',ROOT/'W12-POST/run.py']
    actual={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    from src.post_w12.reporting import canonical_implementation
    return canonical_implementation(actual)

def pilot(c,parent,s):
    p=ROOT/'results/week12/attribution/ATTRIBUTION_QUERY_FREEZE.json'
    if sha(p)!=ATTRIBUTION_SHA:raise ValueError('Original K/L cohort changed; do not resample')
    selected=read(p)['selected'];out={}
    for d in DATASETS:
        metadata={r['query_id']:r for r in s['datasets'][d]['VAL']}
        ids=[r['query_id'] for r in selected[d]]
        if len(ids)!=50 or len(set(ids))!=50 or not set(ids)<=metadata.keys():raise ValueError('Pilot membership mismatch')
        if set(ids)&{r['query_id'] for r in s['datasets'][d]['CONFIRM']}:raise ValueError('Confirmation access blocked')
        out[d]=[dict(query_id=q,paper_id=metadata[q]['paper_id'],group=metadata[q]['group']) for q in ids]
    obj=dict(label=LABEL,stage='pilot',selected=out,source_sha256=ATTRIBUTION_SHA,split_hash=SPLIT_HASH)
    save(output(c)/'PILOT_COHORT.json',obj);return obj

def preregister(c):
    parent,f,s=verify(c);cohort=pilot(c,parent,s)
    ident=dict(label=LABEL,version=c['version'],config=c,implementation=implementation(),
               parent_freeze_sha=FREEZE_SHA,parent_identity=f['identity'],split_hash=SPLIT_HASH,
               cohort_source_sha=ATTRIBUTION_SHA,parser_sha=f['parser_hash'],prompt_sha=f['prompt_source_hash'],
               reader=parent['models']['Qwen'],tokenizer=parent['models']['Qwen'],retriever=parent['models']['SFR'],
               serializer='additive-token-id-blocks-v1',precision='float16',decoding=parent['generation'])
    obj=dict(identity=ident,identity_hash=digest(ident),pilot_cohort_hash=digest(cohort),
             formulation='Rel + eta*sum positive max-aspect-coverage gains - lambda*max cosine redundancy',
             novelty='1-Red diagnostic only; redundant reward omitted; need disabled',
             expansion=c['expansion'],
             selection_outcomes_accessed=False,stage2_requires_promising_and_manual_opt_in=True)
    save(output(c)/'PRECALIBRATION_PROTOCOL.json',obj);return parent,s,cohort,obj

def freeze(c):
    parent,s,cohort,draft=preregister(c)
    path=output(c)/'calibration/CALIBRATION.json'
    if not path.exists():raise ValueError('Allocation-only calibration required before final freeze/pilot')
    cal=read(path)
    if cal['protocol_hash']!=draft['identity_hash'] or cal['status']!='READY':
        raise ValueError('Calibration pending, incompatible, or needs review; pilot blocked')
    if cal['answer_generations']!=0 or cal['quality_metrics_used']:
        raise ValueError('Calibration must be outcome-free')
    test_path=output(c)/'calibration/TESTS.json'
    if not test_path.exists():raise ValueError('Successful pre-calibration test receipt required')
    test=read(test_path)
    if not test['passed'] or test['protocol_hash']!=draft['identity_hash']:
        raise ValueError('Tests must pass for this exact preregistration before final freeze')
    ident=dict(draft['identity'],parameters=cal['parameters'],calibration_sha=sha(path),test_receipt_sha=sha(test_path))
    obj=dict(draft,identity=ident,identity_hash=digest(ident),input_identity=draft['identity'])
    save(output(c)/'BEA_QBGF_V1_FREEZE.json',obj)
    return parent,s,cohort,obj

class Store:
    def __init__(self,path,ident):self.path=Path(path);self.ident=ident
    def file(self,key):return self.path/(digest(key)+'.json')
    def get(self,key):
        p=self.file(key)
        if not p.exists():return None
        r=read(p)
        if r['key']!=key or r['identity_hash']!=digest(self.ident) or r['checksum']!=digest(r['row']):
            raise ValueError('Incompatible/corrupt local result; use a new output version: '+str(p))
        return r['row']
    def put(self,key,row):
        save(self.file(key),dict(label=LABEL,key=key,identity_hash=digest(self.ident),checksum=digest(row),row=row));return row

def expanded(c,s,cohort,decision):
    if decision['decision']!='PROMISING':raise ValueError('Expanded VAL blocked: pilot is not PROMISING')
    out={}
    for d in DATASETS:
        old=cohort['selected'][d];ids={r['query_id'] for r in old}
        candidates=sorted([r for r in s['datasets'][d]['VAL'] if r['query_id'] not in ids],key=lambda r:digest([42,d,r['query_id']]))
        wanted=min(c['expanded_count_per_dataset'],len(s['datasets'][d]['VAL']))
        out[d]=old+[dict(query_id=r['query_id'],paper_id=r['paper_id'],group=r['group']) for r in candidates[:wanted-len(old)]]
    obj=dict(label='POST_W12_EXPLORATORY_VAL',selected=out,includes_all_pilot=True,split_hash=SPLIT_HASH,selection='seed42 SHA256 of dataset/query ID only')
    save(output(c)/'expanded/EXPANDED_COHORT.json',obj);return obj

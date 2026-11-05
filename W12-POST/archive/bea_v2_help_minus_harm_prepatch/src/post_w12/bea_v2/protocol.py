"""Independent immutable artifacts. No writes to historical namespaces."""
import json,hashlib,os,tempfile,subprocess,fcntl,functools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
LABEL='POST_W12_EXPLORATORY_BEA_V2'
DATASETS=['HotpotQA','QASPER','SciFact']
FEATURES=['query_sentence_cosine','passage_query_cosine','retrieval_rank','sentence_rank_within_parent',
          'sentence_position_normalized','sentence_token_cost','passage_token_cost','number_of_sentences_in_parent',
          'budget_label','candidate_pool_width','candidate_sentence_mean_similarity_to_pool','candidate_sentence_max_similarity_to_other_candidates']

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load():
    c=read(ROOT/'W12-POST/bea_v2_config.json')
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='gagan':raise ValueError('Use gagan')
    if c['label']!=LABEL or c['budgets']!=[256,1024] or c['folds']!=5:raise ValueError('Unexpected V2 registry')
    if c['cache']['overwrite'] or c['cache']['force_regenerate'] or not c['cache']['resume']:raise ValueError('Use immutable resume; choose a new version instead')
    return c

def root(c):return ROOT/c['output_root']
def old(c):return ROOT/c['v1_root']
def write(path,obj,replace=False):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'results/post_w12/bea_v2'):raise ValueError('Historical writes prohibited')
    path.parent.mkdir(parents=True,exist_ok=True);value=json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists() and not replace:
        if path.read_text()!=value:raise ValueError('Incompatible existing V2 artifact; use new output version: '+str(path))
        return obj
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'w') as f:f.write(value);f.flush();os.fsync(f.fileno())
        if replace:os.replace(tmp,path)
        else:
            try:os.link(tmp,path)
            except FileExistsError:
                if path.read_text()!=value:raise ValueError('Concurrent incompatible artifact')
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return obj

def text(path,value):
    path=Path(path)
    if not path.resolve().is_relative_to(ROOT/'results/post_w12/bea_v2'):raise ValueError('Historical writes prohibited')
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial');tmp.write_text(value);os.replace(tmp,path)

def envelope(path,identity=None):
    v=read(path)
    if v['checksum']!=digest(v['row']) or (identity and v['identity_hash']!=identity):raise ValueError('Invalid cache identity/checksum: '+str(path))
    return v

def legacy(c,folder,key,identity=None):
    path=old(c)/folder/(digest(key)+'.json')
    if not path.exists():raise ValueError('Required historical artifact missing; restore exact frozen row, never approximate: '+str(key))
    v=envelope(path,identity)
    if v['key']!=key:raise ValueError('Wrong historical key')
    return v['row']

class Store:
    def __init__(self,path,identity):self.path=Path(path);self.identity=identity
    def file(self,key):return self.path/(digest(key)+'.json')
    def get(self,key):
        p=self.file(key)
        if not p.exists():return None
        v=envelope(p,digest(self.identity))
        if v['key']!=key:raise ValueError('Wrong V2 cache key')
        return v['row']
    def put(self,key,row):
        write(self.file(key),dict(key=key,row=row,checksum=digest(row),identity_hash=digest(self.identity),label=LABEL));return row

def exclusive(fn):
    @functools.wraps(fn)
    def wrapper(c=None,*a,**kw):
        c=c or load();root(c).mkdir(parents=True,exist_ok=True)
        with (root(c)/'.lock').open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another V2 stage is active')
            return fn(c,*a,**kw)
    return wrapper

def dependencies(c):
    f=read(old(c)/'BEA_QBGF_V1_FREEZE.json')
    if digest(f['identity'])!=f['identity_hash']:raise ValueError('Invalid V1 freeze')
    from src.post_w12.bea_qbgf.protocol import verify
    parent,main,s=verify(f['identity']['config']) # read-only original science/data/package guard
    for name in ['src/post_w12/bea_qbgf/allocator.py','src/post_w12/bea_qbgf/resources.py']:
        if sha(ROOT/name)!=f['identity']['implementation'][name]:raise ValueError('Frozen V1 computational source changed: '+name)
    if main['identity']!=f['identity']['parent_identity']:raise ValueError('Parent identity changed')
    cohort=read(old(c)/'PILOT_COHORT.json')
    if digest(cohort)!=f['pilot_cohort_hash']:raise ValueError('V1 cohort changed')
    for d in DATASETS:
        members=cohort['selected'][d];meta={r['query_id']:r for r in s['datasets'][d]['VAL']}
        if len(members)!=50 or len({r['query_id'] for r in members})!=50:raise ValueError('Wrong cohort')
        for r in members:
            if r['query_id'] not in meta or r['group']!=meta[r['query_id']]['group']:raise ValueError('Wrong VAL group')
        if {r['query_id'] for r in members}&{r['query_id'] for r in s['datasets'][d]['CONFIRM']}:raise ValueError('Confirmation overlap')
    return parent,f,s,cohort

def implementation():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'W12-POST/bea_v2_config.json',ROOT/'W12-POST/run_bea_v2.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def freeze(c):
    parent,v1,s,cohort=dependencies(c)
    cal=read(root(c)/'calibration/CALIBRATION.json');model=read(root(c)/'utility_model/MODELS.json')
    if cal['status']!='READY' or cal['answer_generations']!=0:raise ValueError('Allocation calibration not ready')
    if cal['model_manifest_sha']!=sha(root(c)/'utility_model/MODELS.json'):raise ValueError('Predictors changed after calibration')
    from .models import verify_models
    verify_models(c)
    for name,h in model['training_sources'].items():
        if sha(ROOT/name)!=h:raise ValueError('Training source changed: '+name)
    ident=dict(label=LABEL,study=c['study'],config=c,implementation=implementation(),
               parent_identity=v1['identity']['parent_identity'],v1_freeze_sha=sha(old(c)/'BEA_QBGF_V1_FREEZE.json'),
               cohort_sha=sha(old(c)/'PILOT_COHORT.json'),model_manifest_sha=sha(root(c)/'utility_model/MODELS.json'),
               feature_schema_sha=sha(root(c)/'FEATURE_SCHEMA.json'),fold_sha=sha(root(c)/'CROSSFIT_FOLDS.json'),
               calibration_sha=sha(root(c)/'calibration/CALIBRATION.json'),threshold=cal['threshold'],
               serializer=v1['identity']['serializer'],reader=v1['identity']['reader'],tokenizer=v1['identity']['tokenizer'],
               parser_sha=v1['identity']['parser_sha'],prompt_sha=v1['identity']['prompt_sha'],decoding=v1['identity']['decoding'])
    obj=dict(identity=ident,identity_hash=digest(ident),exploratory=True,not_fresh_validation=True)
    write(root(c)/'BEA_V2_FREEZE.json',obj);return parent,v1,s,cohort,obj

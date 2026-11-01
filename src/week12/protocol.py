import contextlib,fcntl,hashlib,json,os,subprocess
from pathlib import Path
from datetime import datetime,timezone
import yaml
ROOT=Path(__file__).resolve().parents[2]
PHASES='0ABCDEFGHI'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def guard(dataset=None,split=None,path=None):
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()!='gagan':raise RuntimeError('Week 12 requires gagan')
    if split and split.lower()=='test':raise ValueError('Week12 never evaluates TEST; QASPER TEST permanently closed')
    if dataset=='QASPER' and ((split and split.lower()=='test') or (path and 'test' in Path(path).name.lower())):raise ValueError('QASPER TEST permanently blocked before any read')
def atomic(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n');os.replace(tmp,path)
def immutable(path,obj):
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        if old!=obj:raise ValueError(f'Immutable artifact changed: {path}; choose a new output version')
        return old
    atomic(path,obj);return obj
@contextlib.contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def load(path=None):
    guard();c=yaml.safe_load(Path(path or ROOT/'configs/week12/protocol.yaml').read_text())
    if not (ROOT/c['output_root']).resolve().is_relative_to(ROOT/'results/week12'):raise ValueError('Week12 outputs only')
    if c['labels']!=c['datasets']['SciFact']['labels']:raise ValueError('SciFact labels must match normalized adapter registry')
    if c['candidate_width']!=10 or c['budgets']!=[256,1024]:raise ValueError('N and primary budgets preregistered; use a new protocol for changes')
    if c['tau_p']!=5 or c['tau_s']!=5:raise ValueError('Temperatures frozen at five')
    return c

_SHA_CACHE={}
def cached_sha(path):
    p=Path(path);st=p.stat();key=(str(p),st.st_size,st.st_mtime_ns)
    if key not in _SHA_CACHE:_SHA_CACHE[key]=sha(p)
    return _SHA_CACHE[key]

def code_hash():
    paths=sorted(list((ROOT/'src/week12').glob('*.py'))+[ROOT/'scripts/week12/run.py',ROOT/'scripts/week13/run_confirmation.py',ROOT/'scripts/week12/llmlingua_worker.py']) # notebook builders, validators and smoke tooling do not alter experiment identity
    deps=['src/week10/root_cause/encoding.py','src/week10/root_cause/runtime.py','src/week11/hierarchy.py','src/week11/propositions.py','src/week11/evaluation.py','src/week11/datasets.py','src/week11/qasper_adapter.py','src/week11/readers.py','src/week11/budgets.py','src/week9/senttopk_e3/sentences.py','src/week6/evaluation.py','src/week2/precompute_dev_embeddings.py','src/week2/sweep_runner.py','src/week9/final_h05/historical_backend.py','src/week9/final_h05/literal.py','Baselines/SARA-main/src/model/loader.py']
    paths+= [ROOT/x for x in deps]
    return digest({str(p.relative_to(ROOT)):sha(p) for p in paths})
def identity(cfg,extra=None):
    import importlib.metadata
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','peft','numpy']}
    inputs={}
    for dataset,spec in cfg['datasets'].items():
        for k in ['path','raw_path','corpus_path']:
            if k in spec:
                p=ROOT/spec[k];guard(dataset,spec.get('split'),p)
                if p.exists():inputs[str(p)]=cached_sha(p)
    cohort_path=root(cfg)/'protocol/cohorts.json'
    if cohort_path.exists():inputs['W12_cohorts']=cached_sha(cohort_path)
    split_path=root(cfg)/'protocol/W12_SPLIT_FREEZE.json'
    if split_path.exists():inputs['W12_split_freeze']=cached_sha(split_path)
    compressor_lock=root(cfg)/'protocol/llmlingua_qualification_lock.json'
    if compressor_lock.exists():inputs['LLMLingua_qualification_lock']=cached_sha(compressor_lock)
    return dict(config_hash=digest(cfg),code_hash=code_hash(),models=cfg['models'],packages=versions,source_files=inputs,extra=extra or {})
def root(cfg):return ROOT/cfg['output_root']
class Store:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.identity=identity
        immutable(self.path/'identity.json',identity)
    def get(self,key):
        p=self.path/(digest(key)+'.json')
        if not p.exists():return None
        v=json.loads(p.read_text())
        if v['key']!=key or v['identity']!=digest(self.identity) or v['checksum']!=digest(v['row']):raise ValueError('Cache checksum/identity mismatch')
        return v['row']
    def put(self,key,row):
        old=self.get(key)
        if old is not None:
            if old!=row:raise ValueError('Refusing overwrite of completed result')
            return old
        immutable(self.path/(digest(key)+'.json'),dict(key=key,identity=digest(self.identity),row=row,checksum=digest(row)));return row
    def rows(self):return [self.get(json.loads(p.read_text())['key']) for p in sorted(self.path.glob('*.json')) if p.name!='identity.json']

def authorize(full=False,smoke=True,explicit=False,limit=3):
    guard()
    if full and (smoke or not explicit):raise ValueError('Manual full opt-in requires RUN_FULL=True and RUN_SMOKE=False')
    if smoke and not 1<=limit<=3:raise ValueError('Real/synthetic smoke maximum is three queries TOTAL')
def cohort(cfg,kind='development'):
    if kind=='confirmation':raise RuntimeError('Week13 confirmation locked: Week12 never evaluates confirmation data')
    if cfg.get('split_protocol')=='TRAIN_VAL_CONFIRM_v2':
        from .splits import read_freeze
        if kind not in ('development','validation','train'):raise ValueError('Unknown data role')
        role='TRAIN' if kind=='train' else 'VAL'
        return {d:p[role] for d,p in read_freeze(cfg)['datasets'].items()}
    p=root(cfg)/'protocol'/'cohorts.json'
    if not p.exists():raise RuntimeError('Run notebook 0 first')
    obj=json.loads(p.read_text());assert obj['protocol_hash']==digest(cfg)
    return obj[kind]
def require_full(cfg,phase):
    c=cohort(cfg)
    if not c:raise RuntimeError('No audited development cohort')
    if cfg.get('split_protocol')=='TRAIN_VAL_CONFIRM_v2':
        from .splits import read_freeze
        read_freeze(cfg)
    else:
        freeze=json.loads((root(cfg)/'protocol'/'W12_CONFIRMATION_FREEZE.json').read_text())
        if freeze['protocol_hash']!=digest(cfg):raise ValueError('Protocol changed after cohort freeze')
    if phase=='G' and not (root(cfg)/'lora'/'complete.json').exists():raise RuntimeError('Full LoRA training not complete; base results stay usable')
    return c

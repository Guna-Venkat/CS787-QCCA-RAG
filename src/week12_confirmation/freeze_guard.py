"""Metadata-only fixed-artifact checks and explicit confirmation access capability."""
import json,subprocess,shutil
from dataclasses import dataclass
from pathlib import Path
from src.week12.protocol import ROOT,root,load,identity,digest,sha,immutable,guard
FREEZE_SHA='d0188a3582d6aead0a3a31f9f15bad9b51a88a6b475b05ac79ae38e58f05366b'
SPLIT_SHA='78f5ace11f9080d05dda12716e634020711182f721ac562dc81d19bad6c1c117'
SOURCE_COMMIT='93992932e29252bcedbc56bda02e0c8ae3926001'
METHODS=['NONE','RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL']
BUDGETS=[256,1024]
COUNTS={'HotpotQA':(200,200),'SciFact':(50,43),'QASPER':(0,0)}
VERSION='W12-M-v1'

def folder(synthetic=False):return ROOT/'results/week12/final_confirmation'/('smoke/'+code_hash()[:12] if synthetic else 'v1')
def code_hash():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'scripts/week12/run_final_confirmation.py']
    return digest({str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
def registry():
    return [dict(id=f'{d}:{b}:FIXED_SENTENCE-{m}',dataset=d,budget=b,left='FIXED_SENTENCE',right=m,metric='f1' if d=='HotpotQA' else 'macro_f1',family='primary' if m in ('NONE','RAW_SKIP') else 'secondary') for d in ['HotpotQA','SciFact'] for m in ['NONE','RAW_SKIP','QBGF_FULL'] for b in BUDGETS]

def validate_metadata(f,s):
    if f['split_hash']!=SPLIT_SHA or s['split_hash']!=SPLIT_SHA:raise ValueError('Split hash differs')
    if f['git_commit']!=SOURCE_COMMIT:raise ValueError('Part-I source implementation commit differs')
    if f['method']!='FIXED_SENTENCE' or f['retain_lora'] or f['reader']!='BASE_FP16' or f['retriever']!='SFR' or f['candidate_width']!=10:raise ValueError('Frozen primary system differs')
    if f['budgets']!=BUDGETS or f['confirmation_methods']!=[[m,'BASE_FP16'] for m in METHODS]:raise ValueError('Frozen method/budget registry differs')
    for d,(n,g) in COUNTS.items():
        p=s['datasets'][d];rows=p['CONFIRM'];ids=[r['query_id'] for r in rows]
        if len(rows)!=n or len(set(ids))!=n or len({r['group'] for r in rows})!=g or f['confirmation_ids'][d]!=ids:raise ValueError('Confirmation membership/count differs: '+d)
        for role in ['TRAIN','VAL']:
            for field in ['query_id','group','question_hash']:
                if {r[field] for r in rows}&{r[field] for r in p[role]}:raise ValueError('Confirmation overlap: '+d+'/'+field)
    return True

def verify(cfg=None):
    cfg=cfg or load();guard();a=ROOT/'results/week12/W13_FINAL_METHOD_FREEZE.json';b=root(cfg)/'W13_FINAL_METHOD_FREEZE.json'
    if a.read_bytes()!=b.read_bytes() or sha(a)!=FREEZE_SHA:raise ValueError('Part-I artifacts differ or SHA does not match')
    from src.week12.confirmation import final_freeze
    f,s,h=final_freeze(cfg) # metadata/source checksums only; no example/label loader
    validate_metadata(f,s)
    if f['identity']!=identity(cfg):raise ValueError('Model/scientific identity differs')
    if f['review_source_hash']!=sha(ROOT/'src/week12_final_review.py'):raise ValueError('Reviewed Part-I implementation changed')
    if f['prompt_source_hash']!=sha(ROOT/'src/week12/reader.py') or f['parser_hash']!=sha(ROOT/'src/week12/reader.py'):raise ValueError('Frozen prompt/parser changed')
    return f,s

def protocol(cfg=None,synthetic=False):
    cfg=cfg or load()
    if not synthetic:verify(cfg)
    return dict(artifact_version=VERSION,part='Week12 M',freeze_sha=FREEZE_SHA,split_sha=SPLIT_SHA,source_commit=SOURCE_COMMIT,parent_identity=identity(cfg),M_code_hash=code_hash(),methods=METHODS,budgets=BUDGETS,counts={d:list(c) for d,c in COUNTS.items()},logical_rows=2500,max_distinct_generations=2250,registry=registry(),statistics=dict(resamples=10000,seed=42,primary_holm_family=8,secondary_holm_family=4,H3='descriptive',VAL_confirmation_shift='unpaired descriptive only'),synthetic=synthetic,attribution_used=False)

def preflight(cfg=None,persist=True):
    cfg=cfg or load();f,s=verify(cfg);obj=protocol(cfg);out=folder();dirty=subprocess.check_output(['git','status','--porcelain=v1','-uall'],cwd=ROOT,text=True).splitlines()
    if persist:
        immutable(out/'protocol.json',obj) # written before any confirmation outcome access
        immutable(out/'contrast_registry.json',registry())
    return dict(primary_method=f['method'],reader=f['reader'],methods=METHODS,budgets=BUDGETS,cohort_counts={d:list(c) for d,c in COUNTS.items()},freeze_sha=FREEZE_SHA,split_sha=SPLIT_SHA,confirmation_ids_hash=f['confirmation_ids_hash'],logical_rows=2500,generation_upper_bound=2250,free_disk_GiB=shutil.disk_usage(ROOT).free/1024**3,git_branch='gagan',dirty_worktree=dirty,dirty_policy='All frozen scientific dependencies are verified against Part I, including protected source/config/model identities. Other dirty outputs/notebooks are explicitly listed and do not define Part M.',label_access=False,confirmation_outcomes_accessed=False)

@dataclass(frozen=True)
class Permit:
    freeze_sha:str
    full:bool
    allow_labels:bool
    analysis:bool

def authorize(full=False,allow_labels=False,analysis=False,ack_sha=None):
    if not(full and allow_labels and analysis and ack_sha==FREEZE_SHA):raise PermissionError('Real confirmation needs explicit RUN_FULL, ALLOW_CONFIRMATION_LABEL_ACCESS, RUN_CONFIRMATION_ANALYSIS and exact freeze SHA acknowledgement')
    return Permit(FREEZE_SHA,True,True,True)

def require(permit):
    if not isinstance(permit,Permit) or permit!=Permit(FREEZE_SHA,True,True,True):raise PermissionError('Confirmation data/scoring is inaccessible without explicit full-run permit')

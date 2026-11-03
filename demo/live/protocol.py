"""Demo-only controlled-comparison identities; no historical algorithm changes."""
import hashlib,json
from pathlib import Path
from .resources import configuration
NATIVE='pipeline_native'
CONTROLLED='controlled_sfr'
SCORING_MODES=(NATIVE,CONTROLLED)
VERSION='live-correctness-v2'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def scorer_identity(mode,retriever):
    if mode not in SCORING_MODES:raise ValueError('Unknown scoring mode')
    if mode==NATIVE and retriever=='BM25':return dict(kind='BM25 lexical native',version='project-bm25-v1')
    return dict(kind='SFR cosine',model=configuration()['models']['SFR'],query_role='canonical SFR instruction',document_role='unprefixed',pooling='project window mean; cosine',precision='float16')
def invariants(query,passages,method,budget,segmentation,mode,device,retriever="SFR"):
    c=configuration();reader=Path(__file__).resolve().parents[2]/'src/week12/reader.py'
    return dict(version=VERSION,result_type='LIVE_CUSTOM',query=query,source_hash=digest(passages),segmentation=segmentation,downstream_scorer=scorer_identity(mode,retriever),scoring_mode=mode,allocation_method=method,budget=budget,reader=c['models']['Qwen'],precision='float16',device=device,prompt_parser_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),decoding=c['generation'],seed=c['seed'])
def assert_controlled_pair(a,b):
    if a['scoring_mode']!=CONTROLLED or b['scoring_mode']!=CONTROLLED:raise ValueError('Pair is not controlled')
    if a['retriever']==b['retriever']:raise ValueError('Need distinct candidate retrievers')
    if a['invariants']!=b['invariants']:raise ValueError('Downstream comparison invariants differ')
    if a['scorer_identity']!=b['scorer_identity']:raise ValueError('Downstream scorer differs')
    return True

def validity(result):
    if result.get('result_type')!='LIVE_CUSTOM' or result.get('protocol_version')!=VERSION:
        return 'PRE-FIX LIVE DEMO · OBSOLETE FOR GRANULARITY INTERPRETATION' if result.get('method') in ('FIXED_SENTENCE','QBGF_FULL') else 'LEGACY LIVE RESULT · protocol provenance missing; rerun before comparison'
    if result.get('segmentation') is None or result.get('scorer_identity') is None:return 'INCOMPLETE LIVE PROVENANCE · rerun'
    return None

def archive_obsolete(results,archive):
    for key,r in list(results.items()):
        reason=validity(r)
        if reason:
            r['validity_note']=reason;r['invalid_for_granularity']=r.get('method') in ('FIXED_SENTENCE','QBGF_FULL')
            archive.setdefault(str(key),r)
            # Kept in the result cache, but never accepted as reusable by the UI.
    return archive

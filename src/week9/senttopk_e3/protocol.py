"""Immutable D/E inputs and E3-only identities. Historical packages are read-only."""
import json
from pathlib import Path
from importlib.metadata import version
import yaml
from ..provenance import ROOT,digest,file_hash,source_hashes
from ..caching import read_jsonl,atomic_json
from ..data import resolve
from ..grounded_e import protocol as old_e
from .embeddings import encoder_identity

GRID=[1,2,3,4,6,8]
BASELINE_FIELDS=('question_id','paper_id','policy','prediction','initial_draft','f1','initial_f1','delta_f1','exact_match','help','hurt','answer_changed','final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','retrieval_query_time','selection_latency','extraction_latency','revision_latency','measured_end_to_end_latency')


def sources():
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def config(path):
    cfg=yaml.safe_load(Path(path).read_text())
    if cfg['experiment']!='E3-SentTopK-v1' or cfg['sentence_k']!=GRID or cfg['seed']!=42 or cfg['smoke_queries']!=2:
        raise ValueError('Preregistered E3 v1 grid/seed/smoke sample changed')
    if cfg['encoder']!='sentence-transformers/all-MiniLM-L6-v2' or cfg['encoder_device']!='cpu':raise ValueError('Frozen MiniLM on CPU required')
    if cfg['draft_conditioned'] or cfg['revision_mode'] is not False or cfg['max_new_tokens']!=128:raise ValueError('Primary E3 is query-only; D answer behavior stays unchanged')
    if cfg['quality_tolerance']!=.003 or cfg['epsilon']!=.01:raise ValueError('Preregistered tolerance/epsilon changed')
    return cfg


def output(cfg):return resolve(cfg['output_root'])


def validate_old_e_row(row,chunks,paper_id,draft):
    ids=[c['chunk_id'] for c in chunks]
    if (row['paper_id']!=paper_id or row['initial_draft']!=draft or row['source_selected_chunk_ids']!=ids or
        row['selected_chunk_ids']!=ids or row['source_chunk_hashes']!={c['chunk_id']:digest(c['text']) for c in chunks}):
        raise ValueError('E3/D/E query/paper/chunk identity or text hash mismatch')


def inputs(cfg):
    historical_cfg=old_e.config(resolve(cfg['old_e_config']))
    old_root=old_e.output_dir(historical_cfg)
    if not (old_root/'protocol/d_input_freeze.json').is_file():raise ValueError('Existing immutable E D-input seal is required; E3 may not create it')
    manifest=json.loads((old_root/'fit/manifest.json').read_text())
    if not manifest['complete'] or not manifest['full_fit'] or manifest['stage']!='full':raise ValueError('Completed full FIT E results required')
    old_hashes={str(p.resolve()):file_hash(p) for p in sorted(old_root.rglob('*')) if p.is_file() and p.suffix!='.lock'}
    bundle=old_e.inputs(historical_cfg)  # existing seal verified; never rewritten
    meta=json.loads((old_root/'fit/grounded_e_results.identity.json').read_text())
    expected=dict(stage='full',protocol=old_e.identity(historical_cfg,bundle))
    if meta!=expected or manifest['identity_hash']!=digest(expected['protocol']):raise ValueError('Historical E protocol/source/config mismatch')
    rows=read_jsonl(old_root/'fit/grounded_e_results.jsonl')
    records=bundle['records'];by_query={r['inference']['question_id']:r for r in records}
    grid=set();baselines=[];e_lineage={}
    for row in rows:
        qid=row['question_id'];key=(qid,row['policy'])
        if key in grid or qid not in by_query:raise ValueError('Duplicate/unknown historical E query')
        grid.add(key)
        e_lineage.setdefault(qid,{})[row['policy']]=row['row_content_hash']
        if row['cache_fingerprint']!=digest(meta) or row['row_content_hash']!=digest({k:v for k,v in row.items() if k!='row_content_hash'}):raise ValueError('E row checksum/identity mismatch')
        inf=by_query[qid]['inference'];validate_old_e_row(row,bundle['chunks'][qid],inf['paper_id'],inf['initial_draft'])
        baseline={k:row.get(k) for k in BASELINE_FIELDS}
        baseline.update(evidence_tokens=row['evidence']['evidence_tokens'],sentence_k=None)
        baselines.append(baseline)
    if grid!={(q,p) for q in by_query for p in ('E0','E1-Q','E1-QD','E2')}:raise ValueError('Historical E fixed query/policy grid incomplete')
    for row in bundle['baselines']:
        baselines.append(dict(row,evidence_tokens=0 if row['policy']=='D0' else None,sentence_k=0 if row['policy']=='D0' else None))
    if any(file_hash(p)!=h for p,h in old_hashes.items()):raise ValueError('Historical E artifacts changed while reading')
    bundle['baselines']=baselines
    bundle['e_lineage']=e_lineage
    bundle['historical_cfg']=historical_cfg
    bundle['historical_e_hashes']=old_hashes
    bundle['encoder']=encoder_identity(cfg['encoder'],bundle['seal']['model_hashes'])
    bundle['e3_input_hashes']={**bundle['seal']['input_hashes'],**old_hashes}
    return bundle


def identity(cfg,bundle):
    return dict(schema='E3-SentTopK-v1',config=cfg,sources=sources(),historical_sources=source_hashes(),
        historical_e_sources=old_e.e_sources(),input_hashes=bundle['e3_input_hashes'],d_input_seal_hash=digest(bundle['seal']),
        model_hashes=bundle['seal']['model_hashes'],encoder=bundle['encoder'],
        runtime=dict(torch=version('torch'),transformers=version('transformers'),sentence_transformers=version('sentence-transformers'),dtype='torch.float16',quantization='none',attention_backend='efficient_sdpa'),
        selection_rule='Max FIT F1; within .003 of max choose lower cumulative prompt tokens, then lower final prompt tokens, then smaller k; one fixed configuration.',
        answer_behavior='Unchanged canonical D revision=False, 4 FULL + 6 HEAVY context, trained SARA, greedy max_new_tokens=128. Only evidence text changes.',
        normalization=bundle['encoder']['normalization'],gold_overlap='Not implemented: no prevalidated exact sentence-level gold mapping is available. No gold is used in scoring.')


def smoke_records(records,cfg):
    return sorted(records,key=lambda r:digest(['E3-SentTopK-smoke-v1',cfg['seed'],r['inference']['question_id']]))[:cfg['smoke_queries']]


def pin_protocol(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('E3 protocol changed; use a separate version/output directory')
    else:atomic_json(path,value)

"""Read-only D lineage, separate source identities and immutable E input seals."""
import json
from pathlib import Path
from importlib.metadata import version
from ..provenance import ROOT,digest,file_hash,source_hashes,assert_inference_clean
from ..config import load_config,load_frozen
from ..runner import load_phase,dependencies
from ..data import load_records,resolve
from ..caching import atomic_json
from . import prompts


def config(path):
    cfg=load_config(path)
    if cfg.get('phase')!='e' or cfg.get('grounded_e_version')!='v1' or cfg.get('diagnostic_only') is not True:
        raise ValueError('Use the grounded diagnostic E v1 configuration')
    if cfg['max_new_tokens']!=128 or cfg['extraction_max_tokens']!=256 or cfg['max_evidence_tokens'] not in (64,128):
        raise ValueError('Pinned budgets: answer 128, intermediate extraction 256, evidence 64/128')
    if cfg['preflight_queries']!=3 or cfg['seed']!=42:
        raise ValueError('Preregistered FIT preflight: three deterministic queries, seed 42')
    return cfg


def e_sources():
    return {str(p.relative_to(ROOT)):file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def output_dir(cfg):
    return resolve(cfg['grounded_output_root'])/f"budget_{cfg['max_evidence_tokens']}"


def check_chunks(d1,d2,d3):
    ids=d1['selected_chunk_ids'];chunks=d1['source_chunks']
    if len(ids)!=len(set(ids)) or ids!=[c['chunk_id'] for c in chunks]:
        raise ValueError('D1 source ID/order mismatch')
    for row in (d1,d2,d3):
        if row['selected_chunk_ids']!=ids or [digest(c['text']) for c in row['source_chunks']]!=[digest(c['text']) for c in chunks]:
            raise ValueError('D1/D2/D3 chunk identity/text mismatch; no candidate substitution')
        if row['evidence_k']!=2 or row['specification']['selector']!='DraftSemantic' or row['specification']['revision'] is not False:
            raise ValueError('Controlled D protocol changed')
    assert_inference_clean(chunks)
    return chunks


def inputs(cfg):
    # Existing validators open A-D artifacts read-only. New E files live in a
    # nested package so the historical global A-D source bundle is untouched.
    # A-D pinned the old intermediate extraction budget at 1024; E alone
    # deliberately replaces that budget. Verify upstream under its own protocol.
    upstream_cfg=dict(cfg,extraction_max_tokens=1024)
    frozen=dependencies(upstream_cfg,'d','fit')
    if frozen['a']['settings']['representation']!='FIXED-5' or frozen['a']['settings']['retriever']!='SEMANTIC' or frozen['a']['settings']['retrieval_width']!=20:
        raise ValueError('Frozen A is not FIXED-5 / SEMANTIC @20')
    if frozen['b']['settings']['evidence_k']!=2 or frozen['c']['settings']['selector']!='DraftSemantic':
        raise ValueError('Diagnostic K2/DraftSemantic prerequisites missing')
    primary=ROOT/'results/week9/frozen/w9b_frozen_config.json'
    if load_frozen(primary,'b')['settings']['evidence_k']!=0:
        raise ValueError('Primary K0 must be preserved')
    rows=load_phase(cfg,'d','fit')
    directory=resolve(cfg['output_root'])/'extraction/fit'
    manifest_path=directory/'w9d_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    if not manifest['complete'] or manifest['scope']!='full' or manifest['split']!='fit':
        raise ValueError('D must be complete unrestricted FIT')
    if manifest['source_hashes']!=source_hashes():
        raise ValueError('D source bundle changed')
    if digest(manifest['config'])!=manifest['config_hash']:
        raise ValueError('D manifest config mismatch')
    records,paths=load_records(cfg,'fit')
    grouped={}
    for row in rows:
        key=(row['question_id'],row['policy'])
        if key in grouped:raise ValueError('Duplicate D query/policy')
        grouped[key]=row
    if set(grouped)!={(r['inference']['question_id'],p) for r in records for p in ('D0','D1','D2','D3')}:
        raise ValueError('Incomplete D query-policy coverage')
    chunks={};lineage={};baseline=[]
    baseline_keys=['question_id','paper_id','policy','prediction','initial_draft','f1','initial_f1','delta_f1','exact_match','help','hurt','answer_changed','final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','retrieval_query_time','selection_latency','extraction_latency','revision_latency','measured_end_to_end_latency']
    for record in records:
        inf=record['inference'];qid=inf['question_id']
        group=[grouped[qid,p] for p in ('D0','D1','D2','D3')]
        if any(r['paper_id']!=inf['paper_id'] or r['initial_draft']!=inf['initial_draft'] for r in group) or group[0]['prediction']!=inf['initial_draft']:
            raise ValueError('D canonical draft/paper mismatch')
        chunks[qid]=check_chunks(*group[1:])
        if any(c['paper_id']!=inf['paper_id'] for c in chunks[qid]):raise ValueError('Cross-paper evidence')
        lineage[qid]={r['policy']:r['row_content_hash'] for r in group}
        for r in group:baseline.append({k:r.get(k) for k in baseline_keys})
    input_paths=[directory/'w9d_representation.jsonl',directory/'w9d_representation.identity.json',manifest_path,primary]
    input_paths += [resolve(cfg['output_root'])/'frozen'/f'w9{p}_frozen_config.json' for p in 'abc']
    pinned={str(p.resolve()):file_hash(p) for p in input_paths}
    seal=dict(schema='grounded-e-d-input-seal-v1',split='fit',complete=True,num_queries=len(records),
        input_hashes=pinned,upstream_sources=source_hashes(),model_hashes=frozen['c']['model_hashes'],
        retrieval_config_hash=frozen['a']['settings_hash'],selector_config_hash=frozen['c']['settings_hash'],
        frozen_retrieval={k:frozen['a']['settings'][k] for k in ('representation','retriever','retrieval_width')},
        selector='DraftSemantic',evidence_k=2,primary_policy='K_E=0',
        selected_chunks_digest=digest({q:[dict(id=c['chunk_id'],text_hash=digest(c['text'])) for c in cs] for q,cs in chunks.items()}))
    seal_path=output_dir(cfg)/'protocol/d_input_freeze.json'
    if seal_path.exists():
        if json.loads(seal_path.read_text())!=seal:raise ValueError('Immutable E D-input seal changed')
    else:atomic_json(seal_path,seal)
    return dict(records=records,paths=paths,chunks=chunks,lineage=lineage,baselines=baseline,seal=seal)


def identity(cfg,bundle):
    return dict(schema='grounded-e-v1',config=cfg,config_hash=digest(cfg),e_sources=e_sources(),
        upstream_sources=source_hashes(),input_seal_hash=digest(bundle['seal']),
        input_hashes=bundle['seal']['input_hashes'],model_hashes=bundle['seal']['model_hashes'],
        evidence_budget=cfg['max_evidence_tokens'],extraction_generation_limit=cfg['extraction_max_tokens'],
        prompt_templates=dict(common=prompts.COMMON,draft=prompts.DRAFT,structured=prompts.STRUCTURED,blueprint=prompts.BLUEPRINT,E0=prompts.E0_DEFINITION),
        policy_names=list(prompts.POLICIES),runtime=dict(dtype='torch.float16',quantization='none',attention_backend='efficient_sdpa',torch=version('torch'),transformers=version('transformers')),decoding=dict(do_sample=False,answer_max_tokens=128,extraction_max_tokens=256))


def preflight_records(records,cfg):
    return sorted(records,key=lambda r:digest(['grounded-e-fit-preflight-v1',cfg['seed'],r['inference']['question_id']]))[:cfg['preflight_queries']]

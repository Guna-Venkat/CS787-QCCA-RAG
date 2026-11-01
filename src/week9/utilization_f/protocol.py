"""Read-only E3 lineage, F0 equivalence proof, and separate W9-F identities."""
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
from ..provenance import ROOT, digest, file_hash, assert_inference_clean
from ..caching import read_jsonl, atomic_json
from ..data import resolve
from ..backend import load_tokenizer, checkpoint_hashes, SaraBackend, CachedClient
from ..model_registry import CANONICAL
from ..runner import canonical_context
from ..cost_accounting import QueryCost
from ..senttopk_e3 import protocol as e3
from ..senttopk_e3.runner import validate_grid
from ..senttopk_e3.sentences import validate, evidence
from .prompts import POLICIES, F2_VARIANTS, definitions, messages

SELECTION_RULE = (
    'FIT only: maximum F1, retain candidates within .003. Preserve existing global cost-first '
    'convention: lower cumulative prompt tokens, then materially lower HURT '
    '(at least .01 above minimum eligible HURT is the higher-risk tier), then lower generated '
    'tokens, lower recorded answer-stage latency, shorter template, stable policy name. '
    'D0 is excluded from intervention selection.'
)


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    expected = dict(experiment='W9-F-E3k3-utilization-v1', split='fit', sentence_k=3,
                    policies=list(POLICIES), f2_variants=list(F2_VARIANTS), max_new_tokens=128,
                    seed=42, smoke_queries=2, quality_tolerance=.003, material_hurt_difference=.01,
                    bootstrap_resamples=10000, train_controller=False)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Preregistered FIT-only W9-F configuration changed')
    out = resolve(cfg['output_root']).resolve()
    allowed = (ROOT / 'results/week9/w9f_utilization').resolve()
    if not out.is_relative_to(allowed):
        raise ValueError('W9-F must use its separate result hierarchy')
    return cfg


def pin(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Immutable W9-F identity/freeze changed: {path}')
    else:
        atomic_json(path, value)


def verify_checksums(rows, identity):
    for r in rows:
        if (r.get('cache_fingerprint') != digest(identity) or
                r.get('row_content_hash') != digest({k:v for k,v in r.items() if k!='row_content_hash'})):
            raise ValueError('Saved row checksum/identity mismatch')


def check_evidence(row, record, chunks):
    inf = record['inference']
    if (row['question_id'] != inf['question_id'] or row['paper_id'] != inf['paper_id'] or
            row['question'] != inf['question'] or row['initial_draft'] != inf['initial_draft'] or
            row['sentence_k'] != 3 or row['policy'] != 'E3-k3' or row['coarse_evidence_k'] != 2 or
            row['selected_chunk_ids'] != [c['chunk_id'] for c in chunks] or
            row['source_chunk_hashes'] != {c['chunk_id']:digest(c['text']) for c in chunks}):
        raise ValueError('Frozen E3-k3 query/paper/chunk identity changed')
    validate(row['selected_sentences'], chunks, inf['paper_id'])
    selected = row['selected_sentences']
    ids = [s['sentence_id'] for s in selected]
    if (row['selected_sentence_ids'] != ids or len(ids) != len(set(ids)) or
            row['actual_sentence_k'] != len(ids) or len(ids) > 3 or
            row['evidence_text'] != evidence(selected) or row['evidence_hash'] != digest(row['evidence_text']) or
            selected != sorted(selected, key=lambda s:(s['document_char_start'],s['original_chunk_rank'],s['sentence_order'],s['sentence_id']))):
        raise ValueError('Frozen E3 sentence IDs/text/hashes/presentation changed')
    if row['answer_output_hash'] != digest(row['prediction']):
        raise ValueError('E3 saved answer hash mismatch')
    return dict(question_id=inf['question_id'], paper_id=inf['paper_id'],
                source_row_hash=row['row_content_hash'], evidence_hash=row['evidence_hash'],
                source_chunk_ids=row['selected_chunk_ids'], source_chunk_hashes=row['source_chunk_hashes'],
                selected_sentence_ids=ids, sentence_text_hashes=[s['text_hash'] for s in selected])


def f0_proof(row, prompt, model, tokenizer, calls):
    if (row['model'] != model or row['tokenizer'] != tokenizer or
            row['decoding'] != dict(do_sample=False, max_new_tokens=128) or
            row['final_prompt_hash'] != digest(prompt)):
        raise ValueError('F0 cannot alias E3: generator/tokenizer/decoding/rendered prompt differs')
    if row['selected_sentence_ids']:
        matches = [c for c in calls if c['question_id']==row['question_id'] and
                   c['paper_id']==row['paper_id'] and c['prompt_sha256']==digest(prompt)]
        good = [c for c in matches if c['call_identity']['model']==model and
                c['call_identity']['revision_mode']=='F0' and
                c['call_identity']['decoding_hash']==digest(dict(max_tokens=128,do_sample=False)) and
                c['call_identity']['evidence_ids']==row['selected_sentence_ids'] and
                c['content'].strip()==row['prediction']]
        if not good:
            raise ValueError('F0 alias lacks matching saved E3 LM call')
    return dict(question_id=row['question_id'], paper_id=row['paper_id'], alias='saved E3-k3',
                prompt_hash=digest(prompt), evidence_hash=row['evidence_hash'],
                source_row_hash=row['row_content_hash'], equivalent=True)


def inputs(cfg):
    ecfg = e3.config(resolve(cfg['e3_config']))
    parent = e3.output(ecfg)
    required = ['protocol.json','frozen_e3_senttopk.json','fit/manifest.json',
                'fit/results.jsonl','fit/results.identity.json','fit/lm_calls.jsonl','fit/lm_calls.identity.json']
    if not all((parent / p).is_file() for p in required):
        raise ValueError('Completed E3-k3 inputs required; W9-F cannot create/recompute them')
    bundle = e3.inputs(ecfg)
    identity = e3.identity(ecfg, bundle)
    saved = json.loads((parent/'protocol.json').read_text())
    manifest = json.loads((parent/'fit/manifest.json').read_text())
    freeze = json.loads((parent/'frozen_e3_senttopk.json').read_text())
    if (saved != identity or manifest['protocol_hash'] != digest(identity) or not manifest['complete'] or
            not manifest['full_fit'] or not freeze['frozen'] or freeze['selected_k'] != 3 or
            freeze['selection_split'] != 'fit' or freeze['protocol_identity_hash'] != digest(identity)):
        raise ValueError('E3 protocol/config/sources/freeze/completion mismatch')
    for p, h in freeze['artifact_hashes'].items():
        if file_hash(p) != h:
            raise ValueError(f'Frozen E3 artifact changed: {p}')
    meta = dict(protocol=identity, stage='full')
    for p in ('results','lm_calls'):
        if json.loads((parent/f'fit/{p}.identity.json').read_text()) != meta:
            raise ValueError('E3 row/call identity changed')
    rows = read_jsonl(parent/'fit/results.jsonl'); verify_checksums(rows, meta)
    validate_grid(rows,bundle['records'],bundle['chunks'])
    calls = read_jsonl(parent/'fit/lm_calls.jsonl'); verify_checksums(calls, meta)
    # Index matching calls once. No LM invocation or sentence ranking occurs.
    call_map = {}
    for call in calls:
        call_map.setdefault(call['question_id'],[]).append(call)
    chosen = {r['question_id']:r for r in rows if r['sentence_k']==3}
    if len(chosen) != len(bundle['records']):
        raise ValueError('Missing frozen E3-k3 query')
    import torch
    embeddings = torch.load(bundle['paths']['sfr'],map_location='cpu',weights_only=False)
    if 'embeddings' in embeddings: embeddings=embeddings['embeddings']
    tokenizer = load_tokenizer(bundle['historical_cfg'])
    rendering_backend = SimpleNamespace(tokenizer=tokenizer, answer_max_tokens=128)
    rendering_backend.render = lambda m: SaraBackend.render(rendering_backend,m)
    model = dict(model_id=CANONICAL,model_revision=digest(checkpoint_hashes(bundle['historical_cfg'])),
                 dtype='torch.float16',quantization='none',interface='trained_SARA',attention_backend='efficient_sdpa')
    token_id = dict(name_or_path=tokenizer.name_or_path,tokenizer_class=type(tokenizer).__name__)
    proofs=[];sources=[]
    d0={r['question_id']:r for r in bundle['baselines'] if r['policy']=='D0'}
    for record in bundle['records']:
        inf=record['inference'];q=inf['question_id'];row=chosen[q]
        sources.append(check_evidence(row,record,bundle['chunks'][q]))
        if d0[q]['prediction']!=inf['initial_draft'] or d0[q]['f1']!=row['initial_f1']:
            raise ValueError('D0 draft/score differs from frozen E3 initial answer')
        context,_,nt=canonical_context(record,embeddings,tokenizer,'cpu',torch.float16)
        client=CachedClient(rendering_backend,None,inf,QueryCost(),'revision',context=context)
        prompt=client.prepare_prompt(messages(inf,row['evidence_text'],'F0'))
        if nt!=row['initial_draft_prompt_tokens']:
            raise ValueError('Canonical initial prompt token count changed')
        proofs.append(f0_proof(row,prompt,model,token_id,call_map.get(q,[])))
    # Seal every existing E3 artifact, including inventories/cache files: F never writes them.
    preserved={str(p.resolve()):file_hash(p) for p in parent.rglob('*') if p.is_file() and p.suffix!='.lock'}
    preserved.update(bundle['e3_input_hashes'])
    bundle.update(e3_rows=chosen,e3_protocol=identity,e3_manifest=manifest,
                  evidence_identities=sources,f0_proofs=proofs,tokenizer=tokenizer,
                  embeddings=embeddings,model_identity=model,preserved=preserved,d0=d0)
    return bundle


def identity(cfg,bundle):
    return dict(schema=cfg['experiment'],config=cfg,split='fit',
                sources={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
                prompts=definitions(),selection_rule=SELECTION_RULE,
                upstream_protocol_hash=digest(bundle['e3_protocol']),
                upstream_manifest_hash=file_hash(resolve(e3.config(resolve(cfg['e3_config']))['output_root'])/'fit/manifest.json'),
                input_hashes=bundle['preserved'],model=bundle['model_identity'],
                evidence_identity_hash=digest(bundle['evidence_identities']),
                f0_alias_proof_hash=digest(bundle['f0_proofs']),
                answer_fallback='Empty/invalid wrapper -> exact D0, charged answer call; no extraction fallback.',
                controller='OFF: W9-G must wait for the frozen W9-F intervention action; E3 oracle used F0.')


def unchanged(bundle):
    for path,h in bundle['preserved'].items():
        if file_hash(path)!=h:
            raise ValueError(f'Historical A-E3 artifact changed: {path}')


def smoke_records(records,cfg):
    return sorted(records,key=lambda r:digest(['W9-F-utilization-smoke-v1',cfg['seed'],r['inference']['question_id']]))[:cfg['smoke_queries']]

"""Strict read-only compatibility loader for completed W9-F and E3 FIT outputs."""
import csv
import json
import math
from pathlib import Path
import yaml
from ..provenance import ROOT,digest,file_hash,branch_guard
from ..caching import read_jsonl,atomic_json

ACTIONS=('D0','F0','F2-A')
F_POLICIES=('D0','F0','F1','F2-A','F2-B')


def resolve(path):
    p=Path(path)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def config(path):
    cfg=yaml.safe_load(resolve(path).read_text())
    expected=dict(experiment='W9-G-action-verification-v1',split='fit',expected_queries=1746,
        epsilon=.01,minimum_gate_headroom=.005,minimum_verifier_advantage=.005,
        minimum_three_action_gain=.003,action_preference=['D0','F2-A','F0'],
        outer_folds=5,inner_folds=4,seed=42,quality_tolerance=.003,
        models=['logistic','hist_gradient_boosting'],
        feature_groups=['V0','V1_lexical','V2_length','V3','V4'],
        thresholds=[0.,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95,1.,1.01],run_controller_training=False)
    if any(cfg.get(k)!=v for k,v in expected.items()):
        raise ValueError('Preregistered FIT-only action-verification config changed')
    allowed=(ROOT/'results/week9/w9g_action_verification').resolve()
    if not resolve(cfg['output_root']).is_relative_to(allowed):
        raise ValueError('Refusing to write historical W9-G or upstream result hierarchy')
    if not resolve(cfg['f_root']).is_relative_to(ROOT/'results/week9/w9f_utilization'):
        raise ValueError('Expected completed W9-F utilization artifacts')
    return cfg


def pin(path,value):
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Immutable action-specific identity changed: {path}')
    else:atomic_json(path,value)


def verify_hashes(hashes):
    for p,h in hashes.items():
        if file_hash(resolve(p))!=h:raise ValueError(f'Frozen input/source artifact changed: {p}')


def verify_rows(rows,identity):
    for r in rows:
        if (r['cache_fingerprint']!=digest(identity) or
                r['row_content_hash']!=digest({k:v for k,v in r.items() if k!='row_content_hash'})):
            raise ValueError('Saved action row checksum/identity mismatch')


def aligned(rows,expected_pairs,policies=ACTIONS):
    grouped={p:{} for p in policies}
    expected=dict(expected_pairs)
    if len(expected)!=len(expected_pairs):raise ValueError('Duplicate FIT query in manifest')
    for r in rows:
        p=r['policy'];q=r['question_id']
        if p not in grouped:continue
        if q in grouped[p] or q not in expected or r['paper_id']!=expected[q]:
            raise ValueError('Duplicate/query/paper action alignment failure')
        if not math.isfinite(r['f1']) or not 0<=r['f1']<=1:raise ValueError('Invalid saved FIT score')
        grouped[p][q]=r
    if any(set(g)!=set(expected) for g in grouped.values()):
        raise ValueError('Missing action row or incomplete FIT coverage')
    return grouped


def inputs(cfg):
    froot=resolve(cfg['f_root']);eroot=resolve(cfg['e3_root'])
    required=['protocol.json','config_snapshot.json','prompts.json','source_evidence_identities.json',
        'f0_alias_verification.json','frozen_w9f_f2.json','frozen_w9f_intervention.json',
        'w9f_controller_handoff.json','fit/manifest.json','fit/results.jsonl','fit/results.identity.json']
    if not all((froot/p).is_file() for p in required):raise ValueError('Completed W9-F handoff and full outputs required')
    # No generator/tokenizer/encoder imports or raw DEV/TEST dataset loading.
    fp=json.loads((froot/'protocol.json').read_text());manifest=json.loads((froot/'fit/manifest.json').read_text())
    f2=json.loads((froot/'frozen_w9f_f2.json').read_text());freeze=json.loads((froot/'frozen_w9f_intervention.json').read_text())
    handoff=json.loads((froot/'w9f_controller_handoff.json').read_text())
    if (manifest['split']!='fit' or not manifest['complete'] or not manifest['full_fit'] or
            manifest['queries']!=cfg['expected_queries'] or manifest['rows']!=cfg['expected_queries']*len(F_POLICIES) or
            manifest['protocol_hash']!=digest(fp) or fp['split']!='fit' or
            not f2['frozen'] or f2['split']!='fit' or f2['selected_variant']!='F2-A' or f2['protocol_hash']!=digest(fp) or
            not freeze['frozen'] or freeze['split']!='fit' or freeze['evidence_policy']!='E3-k3' or
            freeze['selected_f2_variant']!='F2-A' or freeze['protocol_hash']!=digest(fp) or
            handoff['intervention_freeze']!=freeze or handoff['split']!='fit' or not handoff['fit_research_labels_only']):
        raise ValueError('F completion/FIT/frozen F2-A/E3-k3/handoff identity mismatch')
    verify_hashes(fp['sources']);verify_hashes(fp['input_hashes']);verify_hashes(freeze['artifact_hashes'])
    if json.loads((froot/'config_snapshot.json').read_text())!=fp['config']:
        raise ValueError('F config snapshot mismatch')
    if json.loads((froot/'prompts.json').read_text())!=fp['prompts'] or f2['prompt']!=fp['prompts']['F2-A']:
        raise ValueError('Frozen F2-A prompt differs from preregistration')
    ep=json.loads((eroot/'protocol.json').read_text());ef=json.loads((eroot/'frozen_e3_senttopk.json').read_text())
    if (ef['selected_k']!=3 or not ef['frozen'] or ef['selection_split']!='fit' or
            ef['protocol_identity_hash']!=digest(ep) or fp['upstream_protocol_hash']!=digest(ep) or
            freeze['source_e3_protocol_hash']!=digest(ep) or
            fp['upstream_manifest_hash']!=file_hash(eroot/'fit/manifest.json') or
            freeze['source_e3_manifest_hash']!=file_hash(eroot/'fit/manifest.json')):
        raise ValueError('F/E3 frozen protocol or k mismatch')
    for key in ('sources','historical_sources','historical_e_sources'):verify_hashes(ep[key])
    verify_hashes(ef['artifact_hashes'])
    expected_meta=dict(protocol=fp,stage='full')
    if json.loads((froot/'fit/results.identity.json').read_text())!=expected_meta:raise ValueError('F result identity mismatch')
    rows=read_jsonl(froot/'fit/results.jsonl');verify_rows(rows,expected_meta)
    with resolve(cfg['fit_manifest']).open() as f:
        pairs=[(r['question_id'],r['paper_id']) for r in csv.DictReader(f)]
    if len(pairs)!=cfg['expected_queries']:raise ValueError('FIT manifest query count differs')
    all_groups=aligned(rows,pairs,F_POLICIES)
    groups={p:all_groups[p] for p in ACTIONS}
    emeta=dict(protocol=ep,stage='full')
    if json.loads((eroot/'fit/results.identity.json').read_text())!=emeta:raise ValueError('E3 row identity mismatch')
    erows=read_jsonl(eroot/'fit/results.jsonl');verify_rows(erows,emeta)
    e3={r['question_id']:r for r in erows if r['sentence_k']==3}
    sources=json.loads((froot/'source_evidence_identities.json').read_text())
    if digest(sources)!=fp['evidence_identity_hash']:raise ValueError('F source evidence identity hash changed')
    source_map={r['question_id']:r for r in sources}
    if len(source_map)!=len(sources) or set(source_map)!=set(dict(pairs)) or set(e3)!=set(source_map):
        raise ValueError('E3/source query coverage mismatch')
    for q,pid in pairs:
        base=groups['D0'][q];src=e3[q];saved=source_map[q]
        if (base['prediction']!=base['initial_draft'] or base['paper_id']!=src['paper_id'] or
                base['prediction']!=src['initial_draft'] or base['f1']!=src['initial_f1']):
            raise ValueError('D0/E3 original draft/score mismatch')
        if (saved['source_row_hash']!=src['row_content_hash'] or saved['paper_id']!=pid or
                saved['evidence_hash']!=src['evidence_hash'] or
                saved['selected_sentence_ids']!=src['selected_sentence_ids'] or
                saved['sentence_text_hashes']!=[s['text_hash'] for s in src['selected_sentences']]):
            raise ValueError('E3-k3 source sentence identity mismatch')
        for policy in ('F0','F2-A'):
            r=groups[policy][q]
            if (r['initial_draft']!=base['prediction'] or r['initial_f1']!=base['f1'] or
                    r['question']!=src['question'] or r['sentence_k']!=3 or
                    r['evidence_text']!=src['evidence_text'] or r['evidence_hash']!=digest(src['evidence_text']) or
                    r['source_e3_row_hash']!=src['row_content_hash'] or
                    r['selected_sentence_ids']!=src['selected_sentence_ids'] or
                    r['sentence_text_hashes']!=[s['text_hash'] for s in src['selected_sentences']] or
                    r['selected_chunk_ids']!=src['selected_chunk_ids'] or r['source_chunk_hashes']!=src['source_chunk_hashes'] or
                    abs(r['delta_f1']-(r['f1']-base['f1']))>1e-12 or
                    r['help']!=(r['delta_f1']>.01) or r['hurt']!=(r['delta_f1']<-.01)):
                raise ValueError('F action/evidence/score identity mismatch')
        f0=groups['F0'][q]
        if f0['prediction']!=src['prediction'] or f0['f1']!=src['f1'] or f0['exact_match']!=src['exact_match']:
            raise ValueError('F0 is not equivalent saved E3-k3')
    hrows=handoff['queries'];hg={r['question_id']:r for r in hrows}
    if len(hg)!=len(hrows) or set(hg)!=set(dict(pairs)):raise ValueError('Handoff query coverage mismatch')
    for q,r in hg.items():
        intervention=all_groups[freeze['actual_policy']][q];base=groups['D0'][q]
        if (r['paper_id']!=base['paper_id'] or r['d0_f1']!=base['f1'] or
                r['intervention_f1']!=intervention['f1'] or r['evidence_hash']!=intervention['evidence_hash'] or
                r['selected_sentence_ids']!=intervention['selected_sentence_ids']):
            raise ValueError('Handoff saved-action consistency failure')
    # Every original F and E3 file is preserved, plus historical G code/results.
    hashes=dict(fp['input_hashes'])
    for root in (froot,eroot,ROOT/'results/week9/controller'):
        if root.exists():hashes.update({str(p.resolve()):file_hash(p) for p in root.rglob('*') if p.is_file() and p.suffix!='.lock'})
    old_g=[ROOT/'src/week9/adaptive_ke.py',ROOT/'configs/week9/w9g.yaml',
        ROOT/'notebooks/week9/week9g_adaptive_ke.ipynb',ROOT/'results/week9/frozen/w9g_frozen_config.json']
    old_g.extend((ROOT/'scripts/week9').glob('run_w9g*.py'))
    hashes.update({str(p):file_hash(p) for p in old_g if p.is_file()})
    hashes[str(resolve(cfg['fit_manifest']))]=file_hash(resolve(cfg['fit_manifest']))
    return dict(groups=groups,e3=e3,pairs=pairs,input_hashes=hashes,
        f_protocol_hash=digest(fp),e3_protocol_hash=digest(ep),f2_variant='F2-A',
        frozen_intervention=freeze['actual_policy'],source_e3_encoder=ep['encoder'])


def identity(cfg,bundle):
    own=('__init__.py','protocol.py','preflight.py','runner.py')
    return dict(schema=cfg['experiment'],config=cfg,split='fit',
        sources={str((Path(__file__).parent/p).relative_to(ROOT)):file_hash(Path(__file__).parent/p) for p in own},
        input_hashes=bundle['input_hashes'],f_protocol_hash=bundle['f_protocol_hash'],
        e3_protocol_hash=bundle['e3_protocol_hash'],frozen_f2_variant='F2-A',
        existing_f_intervention=bundle['frozen_intervention'],
        action_spaces={'PRE-A':['D0','F2-A'],'PRE-B':['D0','F0'],'PRE-C':['D0','F0','F2-A']},
        oracle_ties='Exact maximum: D0 then F2-A then F0. Epsilon .01: first action in the same preference within epsilon of maximum.',
        rules='G1 epsilon headroom >=.005; G2 epsilon headroom >=.005 and advantage over G1 headroom >=.005; G3 exact extra oracle value over D0/F0 >=.003. No arbitrary significance test.',
        cost_semantics='Oracle chosen-action costs exclude searching all actions. Post-generation F0 verification pays F0 even when KEEP; no second-call saving.',
        confidence='Saved generation logprobs/entropy/EOS confidence unavailable; no generation or confidence backfill.',
        training='Disabled; preflight creates no fitted estimator or OOF predictions.')

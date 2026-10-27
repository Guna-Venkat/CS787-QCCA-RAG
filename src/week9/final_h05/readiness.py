"""Publish v2 only after exact recovery, all-adapter checks and safe replay."""
import csv
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from .history import recovery,replay
from .producer import ORDER,dry_run
from .materialize import TEST_FORMATTED,TEST_RAW
from ..final_h import protocol as p
from .storage import atomic_json

V1=p.ROOT/'results/week9/w9h_final/v1'
V2=p.ROOT/'results/week9/w9h_final/v2'
V1_SHA='0df18e811072a72a1fca5fe862a5d90fe1d45a2617c3f34af3142f23eaab4c08'


def sources():
    paths=list((p.ROOT/'src').rglob('*.py'))+list((p.ROOT/'scripts').rglob('*.py'))
    paths+=list((p.ROOT/'Baselines/SARA-main/src').rglob('*.py'))
    paths+=[Path(__file__).parent/'historical/recovery.json']
    return {str(path.relative_to(p.ROOT)):p.file_hash(path) for path in sorted(set(paths))}


def contracts():
    result=p.system_contracts()
    for name in ('D0','semantic_k2','semantic_k4','semantic_k5','semantic_k6','semantic_k8','BM25_k8','original_SARA_G0'):
        result[name]['generator']=dict(result[name]['generator'],dtype='torch.bfloat16',attention_backend='historical_default_sdpa')
    result['BM25_k8']['decoding']=dict(do_sample=False,max_new_tokens=64)
    result['original_SARA_G0']['recovery']=dict(classification='RECOVERED_EXACT',commit=recovery()['commit'],
        source_hashes={k:v['sha256'] for k,v in recovery()['files'].items()})
    for name in (p.PRIMARY,p.SAFE,p.VERIFY):result[name]['initial_pass_precision']='Historical original SARA BF16, as saved W7/W8 first passes; frozen FP16 E3/F revision unchanged'
    return result


def make_registry(source_hashes):
    scientific=contracts();systems={}
    needs={'D0':['TEST formatter','BM25256','SFR BF16 passage vectors'],
        'semantic_k4':['D0 exact alias'], 'D1':['D0','W9-A FIXED-5/SEMANTIC@20'],
        p.PRIMARY:['D0','D1 exact source chunks'],p.SAFE:[p.PRIMARY,'D0'],p.VERIFY:[p.PRIMARY,'D0','audited G2 package'],
        'original_SARA_G0':['D0','SFR BF16 passage vectors','exact recovered W8 gate/selector/generation']}
    for name in ORDER:
        command=['sara_env/bin/python','scripts/week9/h05/produce.py','--root',str(V2.relative_to(p.ROOT)),'--system',name,'--execute']
        systems[name]=dict(producer_id=name,method_family='W8 historical' if name=='original_SARA_G0' else 'W9 frozen' if name in ('D1',p.PRIMARY,p.SAFE,p.VERIFY) else 'canonical static/first pass',
            executable_entrypoint='scripts/week9/h05/produce.py',command=command,source_hashes=source_hashes,
            required_frozen_inputs=needs.get(name,['D0 membership','canonical BM25 top10/SFR']),
            method_config=scientific[name],checkpoint='Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42',
            gpu=name not in ('semantic_k4',p.VERIFY),result_path=str((V2/'predictions'/f'{name}.jsonl').relative_to(p.ROOT)),
            receipt_path=str((V2/'receipts'/f'{name}.json').relative_to(p.ROOT)),
            resume='atomic query shards; exact freeze/config/source identity',
            membership='All queries from frozen TEST input namespace; question/paper IDs asserted at H2, deferred at H0.5',
            expected_output_schema=['system','split','question_id','paper_id','prediction','initial_draft','final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls'],
            conceptual_lm_calls='1 or 2 conditional on original promotion' if name=='original_SARA_G0' else 2 if name in ('D1',p.PRIMARY,p.SAFE,p.VERIFY) else 1,
            conceptual_call_note='D1/F0 zero-evidence fallback pays only initial pass; VERIFY retains F0 cost on KEEP',
            reusable_output='semantic_k4 aliases D0; SAFE/VERIFY reuse exact PRIMARY evidence/candidate; no generation rerun on valid completed shards',
            compatible_test_predictions_already_exist=False,
            cost_accounting='Actual serialized LM prompt sums; conceptual first pass counted; offline SFR/index, cache lookup, stage latency and measured end-to-end remain separate; missing stays null')
    return dict(schema='W9-H05-producers-v2',systems=systems,order=ORDER,
        execution_order_reason='Build shared retrieval/SFR before SARA; reuse D0 and exact D1/E3 intermediates; G2 CPU only; W8 separate final process to avoid duplicate/incompatible Mistral instances.')


def spec(registry):
    return dict(schema='W9-H-materialization-v1',split='test',
        test_inputs=dict(reformatted=str(TEST_FORMATTED),raw=str(TEST_RAW)),
        membership_path=str((V2/'shared/membership.jsonl').relative_to(p.ROOT)),
        labels_path=str((V2/'shared/labels.jsonl').relative_to(p.ROOT)),
        provenance='Inputs and IDs are opened only after matching explicit H2 claim. All outputs are fresh; no TEST metrics at H0.5.',
        systems={name:dict(command=v['command'],producer_source_hashes=v['source_hashes'],predictions_path=v['result_path'],receipt_path=v['receipt_path'],method_config=v['method_config']) for name,v in registry['systems'].items()})


def preflight(staging):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    p.branch_guard();p.require(p.file_hash(V1/'w9h_pretest_freeze_manifest.json')==V1_SHA,'Do not modify v1 freeze')
    staging=Path(staging);staging.mkdir(parents=True,exist_ok=True)
    registry=make_registry(sources());atomic_json(staging/'producer_registry.json',registry)
    proofs=[dry_run(staging,name) for name in ORDER]
    from .synthetic import run as fixture
    with tempfile.TemporaryDirectory(prefix='h05_synthetic_') as temporary:
        synthetic=fixture(temporary)
    w8=replay()
    from .replays import upstream
    non_test=upstream()
    for name in ('adapter_model.safetensors','sara_extras.safetensors'):
        expected=p.read(V1/'w9h_pretest_freeze_manifest.json')['dependency_hashes']
        path=p.ROOT/'Baselines/SARA-main/checkpoints/finetune/sara_qasper_proj_lr5e4_seed42'/name
        p.require(p.file_hash(path)==expected[str(path)],'Frozen checkpoint differs')
    p.require(TEST_FORMATTED.is_file() and TEST_RAW.is_file(),'Explicit TEST input paths missing; no data file has been opened')
    result=dict(complete=True,dry_runs=proofs,synthetic=synthetic,w8_recovery=w8,non_test_replay=non_test,
        test_access=dict(path_existence_only=True,ids_deferred_to_h2=True,test_schema_or_records_opened=False,test_labels_opened=False,test_metrics_computed=False),
        v1_freeze_sha256=p.file_hash(V1/'w9h_pretest_freeze_manifest.json'))
    atomic_json(staging/'preflight.json',result)
    return registry,result


def validate_v2(root):
    root=Path(root)
    p.require(root.resolve()==V2.resolve(),'Final producers require explicit versioned v2 root')
    freeze=p.validate_freeze(root)
    p.require(freeze['schema']=='W9-H-pretest-v2' and freeze['h05_readiness']['passed'],'H0.5 v2 must pass before H2')
    p.require(freeze['w8_recovery']['classification']=='RECOVERED_EXACT','Unverified W8 substitute forbidden')
    p.require(set(freeze['producer_contracts'])==set(p.REQUIRED),'Missing required producer')
    p.require(p.file_hash(V1/'w9h_pretest_freeze_manifest.json')==V1_SHA,'v1 freeze changed')
    p.require(freeze['producer_runtime']==runtime(),'Producer runtime differs from pretest freeze')
    return freeze


def runtime():
    return {name:importlib.metadata.version(name) for name in ('transformers','sentence-transformers','datasets','huggingface-hub','llama-index-core','llama-index-retrievers-bm25','bm25s','peft')}


def compatibility_matrix():
    return [dict(system=name,executable=True,exact_method_recovered=True,
            same_test_membership='ENFORCED_AT_H2_IDS_DEFERRED',f1_em_comparable=True,final_prompt_comparable=True,
            cumulative_prompt_comparable=True,lm_calls_comparable=True,latency_comparable=False,
            pairable_with_primary='CONTRACT_ENFORCED_AT_H2',notes='Actual LM prompts summed. Latency stage-scoped; no common measured end-to-end. No TEST outputs observed.') for name in ORDER]


def publish(config_path):
    p.branch_guard();cfg=p.load_config(config_path)
    p.require(p.resolve(cfg['output_root']).resolve()==V2,'Do not edit v1; publish v2 only')
    if (V2/'w9h_pretest_freeze_manifest.json').exists():return validate_v2(V2)
    p.require(not V2.exists(),'Preserve incomplete existing v2 directory; no in-place recreation')
    staging=p.ROOT/'results/week9/patch_verification/w9h_h05/staging'
    registry,proof=preflight(staging)
    # Validate original immutable dependency chain, including actual backbone weights.
    old=p.validate_freeze(V1,need_ready=False)
    from huggingface_hub import snapshot_download
    sfr_path=Path(snapshot_download('Salesforce/SFR-Embedding-Mistral',local_files_only=True))
    encoder_path=Path(snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2',local_files_only=True))
    extra={}
    for folder in (sfr_path,encoder_path):
        weights=[path for path in folder.rglob('*') if path.is_file() and path.suffix in ('.safetensors','.bin')]
        p.require(bool(weights),'Frozen model weights unavailable: '+str(folder))
        for index in folder.glob('*.index.json'):
            names=set(p.read(index).get('weight_map',{}).values())
            p.require(all((folder/name).is_file() for name in names),'Missing indexed model shard: '+str(folder))
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.safetensors','.bin','.json','.txt','.model'):extra[str(path)]=p.file_hash(path)
        ref=folder.parent.parent/'refs/main'
        if ref.exists():extra[str(ref)]=p.file_hash(ref)
    p.require(any(name.endswith('.safetensors') for name in extra),'Frozen inference model weights missing')
    p.require(sources()==next(iter(registry['systems'].values()))['source_hashes'],'Sources changed during preflight')
    V2.mkdir(parents=True)
    for name,value in [('producer_registry.json',registry),('test_materialization_spec.json',spec(registry)),
                       ('w8_recovered_provenance.json',proof['w8_recovery']),('preflight_validation.json',proof),
                       ('baseline_registry.json',p.read(V1/'baseline_registry.json'))]:p.write_once(V2/name,value)
    protocol=p.read(V1/'protocol.json')
    protocol.update(schema='W9-H-v2',materialization_spec=spec(registry),system_contracts=contracts(),
        producer_registry='producer_registry.json',membership_policy='All IDs and counts deferred to authorized H2; namespace=TEST; exact query/paper coverage required for all 12',
        outcome_independent_metadata_corrections=[
            'BM25 historical source/config cap is 64, not the provisional blanket 128 in v1; preserve historical baseline, no new setting selected.',
            'Original W7/W8 first-pass/static loader default is BF16; W9 E3/F revision remains the already-frozen FP16 backend. Pin both accurately, no checkpoint/prompt/policy change.',
            'Archived W8 sources are isolated and byte-exact; do not use current retired/proxy generation implementation.'])
    p.write_once(V2/'protocol.json',protocol)
    matrix=compatibility_matrix()
    with (V2/'producer_compatibility_matrix.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(matrix[0]));writer.writeheader();writer.writerows(matrix)
    (V2/'producer_compatibility_matrix.csv').chmod(0o444)
    dependency=dict(old['dependency_hashes']);dependency.update(sources());dependency.update(extra)
    for path in V2.iterdir():
        if path.is_file():dependency[str(path)]=p.file_hash(path)
    dependency[str(Path(config_path).resolve())]=p.file_hash(config_path)
    dependency[str(V1/'w9h_pretest_freeze_manifest.json')]=V1_SHA
    for name in ('configs/week2/sweep_config.yaml','configs/week7/w7b_semantic_budget_relearning.yaml'):
        dependency[name]=p.file_hash(p.ROOT/name)
    freeze={k:v for k,v in old.items() if k not in ('manifest_digest','dependency_hashes','blockers','h0_passed')}
    freeze.update(schema='W9-H-pretest-v2',timestamp_utc=p.timestamp(),git_commit=p.commit(),dependency_hashes=dependency,
        blockers=[],h0_passed=True,h05_readiness=dict(passed=True,test_outcomes_used=False,source='preflight_validation.json'),
        producer_contracts=contracts(),w8_recovery=proof['w8_recovery'],sfr=dict(model_id='Salesforce/SFR-Embedding-Mistral',revision=sfr_path.name),
        producer_runtime=runtime(),v1_freeze_sha256=V1_SHA,required_systems=p.REQUIRED,final_confirmation_started=False,
        no_scientific_method_changed=True,actual_final_systems_executed=0)
    freeze['manifest_digest']=p.digest(freeze)
    p.write_once(V2/'w9h_pretest_freeze_manifest.json',freeze)
    p.write_once(V2/'audit_status.json',dict(complete=True,passed=True,blockers=[],w8_provenance='RECOVERED_EXACT',
        all_producers=12,freeze_sha256=p.file_hash(V2/'w9h_pretest_freeze_manifest.json'),test_outcomes_used=False,final_confirmation_started=False))
    from .reporting import write
    write(V2,freeze,proof,registry)
    return freeze

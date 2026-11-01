"""Patch regressions: model/latent separation, identity safety and explicit modes."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.week9.model_registry import MODELS,CANONICAL,assert_projector_compatible,runtime_identity
from src.week9.model_runtime import parse_final,memory_status,TextRuntime
from src.week9.revision import revision_messages,prompt_definition,write_revision_freeze,read_revision_mode,REVISION_INSTRUCTION
from src.week9.provenance import digest,branch_guard
from src.week9.caching import RowCache
from src.week9.backend import CachedClient
from src.week9.cost_accounting import QueryCost
from src.week9.baseline_registry import aligned_rows


def test_branch_is_gagan_and_guard_rejects_main(monkeypatch):
    assert branch_guard()=='gagan'
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'main\n')
    with pytest.raises(RuntimeError):branch_guard()


def test_registry_official_roles_and_latent_boundary():
    assert len(MODELS)==6
    assert len({m.model_id for m in MODELS})==6
    for m in MODELS:
        assert m.roles and m.hidden_size>0
        assert_projector_compatible(m.model_id,'text')
        if m.model_id!=CANONICAL:
            with pytest.raises(ValueError):assert_projector_compatible(m.model_id,'sara_latent')
    assert_projector_compatible(CANONICAL,'sara_latent')
    r=TextRuntime(MODELS[2].model_id,'pinned')
    with pytest.raises(ValueError):r.generate('prompt',retrieval_embeds=object())


def test_runtime_requires_pinned_revisions_and_counts_reasoning_separately():
    with pytest.raises(ValueError):runtime_identity(CANONICAL,'main')
    assert parse_final('<think>private reasoning</think>final',True)==('final',True)
    assert parse_final('<think>unfinished',True)==('',True)
    assert parse_final('reasoning without boundary',True)==('',False)
    assert memory_status(19_999_999_999)=='PASS' and memory_status(20_000_000_000)=='FAIL'


def test_loading_another_runtime_requires_unload(monkeypatch):
    import torch
    a=TextRuntime(CANONICAL,'pinned');b=TextRuntime(MODELS[2].model_id,'pinned')
    monkeypatch.setattr(TextRuntime,'_owner',a)
    with pytest.raises(RuntimeError,match='Release'):b.load()
    a.close();assert TextRuntime._owner is None


def test_f0_f1_prompts_preserved_and_f2_is_distinct():
    inf={'question':'Q','initial_draft':'draft'}
    assert revision_messages(inf,'E',True)==revision_messages(inf,'E','F1')
    assert revision_messages(inf,'E',False)==revision_messages(inf,'E','F0')
    assert revision_messages(inf,'E','F1')[0]['content'].startswith(REVISION_INSTRUCTION)
    assert 'not as an authority' in revision_messages(inf,'E','F2')[0]['content']
    assert len({prompt_definition(m)['prompt_hash'] for m in ('F0','F1','F2')})==3


def test_mode_sidecar_is_fit_frozen_and_settings_matched(tmp_path):
    cfg={'output_root':str(tmp_path)};settings={'revision':'F2'}
    write_revision_freeze(cfg,settings)
    assert read_revision_mode(cfg,{'settings':settings})=='F2'
    with pytest.raises(ValueError):read_revision_mode(cfg,{'settings':{'revision':'F1'}})
    with pytest.raises(FileExistsError):write_revision_freeze(cfg,{'revision':'F1'})


def test_cache_distinguishes_model_mode_evidence_and_task(tmp_path):
    class Backend:
        answer_max_tokens=8
        mode='F0';model='M0'
        def identity(self):return {'model_id':self.model,'model_revision':'pin','dtype':'float16','quantization':'none'}
        def render(self,m):return 'same prompt'
        def generate(self,*args):return dict(content=self.model,prompt_tokens=4,generated_tokens=2,latency=.1)
    backend=Backend();inf={'question_id':'q','paper_id':'p'}
    with RowCache(tmp_path/'calls.jsonl',{'run':1}) as cache:
        for model,mode,stage,ids in [('M0','F0','revision',['a']),('M1','F0','revision',['a']),('M0','F2','revision',['a']),('M0','F0','extraction',['a']),('M0','F0','revision',['b'])]:
            backend.model=model
            client=CachedClient(backend,cache,inf,QueryCost(),stage,evidence_ids=ids,revision_mode=mode)
            assert client.chat([{'role':'user','content':'Q'}])['content']==model
        assert len(cache.rows)==5


def test_baseline_alignment_requires_same_split_ids_papers_and_scorer():
    a=[{'question_id':'q','paper_id':'p','split':'dev','evaluation_hash':'hash'}]
    aligned_rows(a,a)
    for override in [{'split':'internal_val'},{'question_id':'different'},{'paper_id':'other'},{'evaluation_hash':'other'}]:
        with pytest.raises(ValueError):aligned_rows(a,[dict(a[0],**override)])
    with pytest.raises(ValueError):aligned_rows(a+a,a)


def test_f_interventions_expose_all_modes_and_prefixes():
    from src.week9.runner import interventions
    frozen={'b':{'settings':{'evidence_k':2}},'c':{'settings':{'selector':'DraftSemantic'}},'e':{'settings':{'format':'structured'}}}
    specs=interventions('f',{},frozen)
    assert len(specs)==15
    assert {s['revision'] for s in specs}=={'F0','F1','F2'}
    for mode in ('F0','F1','F2'):assert {s['evidence_k'] for s in specs if s['revision']==mode}==set(range(5))


def test_snapshot_requires_all_shards(tmp_path):
    from src.week9.model_preflight import verify_snapshot
    for f in ('config.json','tokenizer_config.json','tokenizer.json'):(tmp_path/f).write_text('{}')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'one.safetensors','b':'two.safetensors'}}))
    (tmp_path/'one.safetensors').write_bytes(b'weights')
    with pytest.raises(FileNotFoundError):verify_snapshot(tmp_path)
    (tmp_path/'two.safetensors').write_bytes(b'weights')
    assert verify_snapshot(tmp_path)==14


def test_g_uses_frozen_f2_supervision(tmp_path,monkeypatch):
    from src.week9 import runner
    cfg={'output_root':str(tmp_path),'adaptive':{}}
    settings={'revision':'F2'};write_revision_freeze(cfg,settings)
    rows=[{'question_id':'q','paper_id':'p','evidence_k':k,'specification':{'revision':mode}} for mode in ('F0','F1','F2') for k in range(5)]
    monkeypatch.setattr(runner,'load_phase',lambda *args:rows)
    observed=[]
    def gate(actual,thresholds):
        observed.extend(actual);return {'proceed':False,'status':'SKIPPED — insufficient adaptive headroom'}
    monkeypatch.setattr(runner,'headroom_gate',gate)
    cfg['_paper_fold']={'role':'train'}
    result=runner.run_g(cfg,'fit',{'f':{'settings':settings}},tmp_path,None)
    assert result['status'].startswith('SKIPPED')
    assert len(observed)==5 and all(r['specification']['revision']=='F2' for r in observed)


def test_invalid_half_precision_scores_are_not_quality_results():
    import torch
    from src.week9.model_runtime import validate_scores
    validate_scores([torch.tensor([[0.,float('-inf'),1.]])])
    for score in [float('nan'),float('inf'),float('-inf')]:
        with pytest.raises(RuntimeError,match='Numerically invalid'):validate_scores([torch.tensor([[score]])])


def test_registered_real_baselines_keep_split_and_exact_imports():
    from src.week9.baseline_registry import saved_directory
    from src.week9.provenance import ROOT,file_hash
    imported=json.loads((ROOT/'results/week9/baselines/w8_import_manifest.json').read_text())
    assert imported['write_branch']=='gagan'
    for name,info in imported['files'].items():assert file_hash(ROOT/info['imported_path'])==info['sha256']
    h=saved_directory('results/week8/w8h_full_system/full')
    assert (h/'original_SARA_predictions.jsonl').exists()
    for split,count in [('internal_val',342),('dev',231)]:
        entries=json.loads((ROOT/f'results/week9/baselines/{split}/baseline_manifest.json').read_text())['entries']
        primary=next(e for e in entries if e['baseline_id']=='original_SARA_G0')
        assert primary['status']=='AVAILABLE' and primary['num_queries']==count
        assert primary['split']==split


def test_model_spec_identity_survives_json_cache_roundtrip():
    for model in MODELS:
        value=model.to_dict()
        assert json.loads(json.dumps(value))==value


def test_complete_commit_snapshot_is_found_without_main_ref(tmp_path):
    from src.week9.model_preflight import find_cached_snapshot
    folder=tmp_path/('models--'+CANONICAL.replace('/','--'))/'snapshots'/'pinned'
    folder.mkdir(parents=True)
    for name in ('config.json','tokenizer_config.json','tokenizer.json'):(folder/name).write_text('{}')
    (folder/'model.safetensors').write_bytes(b'weights')
    path,size=find_cached_snapshot(CANONICAL,tmp_path)
    assert path==folder and size==7


def test_spoofed_label_cannot_inject_alternative_checkpoint(tmp_path):
    from src.week9.backend import SaraBackend
    (tmp_path/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':MODELS[1].model_id}))
    with pytest.raises(ValueError,match='Mistral-specific'):
        SaraBackend({'checkpoint':str(tmp_path),'first_pass_sara_model':CANONICAL})


@pytest.mark.parametrize('evidence, expected, error', [
    ([['first', 'second', 'first', ' ']], ['first', 'second'], None),
    ([['first', 'second'], ['second', 'first']], ['first', 'second'], None),
    ([['first'], ['different']], None, 'conflicting evidence'),
    ([], None, 'alignment missing'),
])
def test_raw_evidence_duplicate_question_safety(monkeypatch, evidence, expected, error):
    from src.week9 import data
    paper = {'qas': {'question': [' Same question? '] * len(evidence),
        'answers': [{'answer': [{'evidence': items}]} for items in evidence]}}
    monkeypatch.setattr(data, 'raw_dataset', lambda path: [paper])
    record = {'inference': {'paper_id': '0', 'question': 'Same question?'}}
    if error:
        with pytest.raises(ValueError, match=error):
            data.evaluation_evidence(record, 'unused')
    else:
        assert data.evaluation_evidence(record, 'unused') == expected

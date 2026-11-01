"""Reader download/setup logic with synthetic caches; no network or GPU calls."""
import importlib.util
import json
from pathlib import Path
import pytest

SCRIPT=Path(__file__).resolve().parents[2]/'scripts/week10/prepare_readers.py'
spec=importlib.util.spec_from_file_location('w10_reader_setup',SCRIPT)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def cache(path):
    path.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','tokenizer_config.json','tokenizer.json'):(path/name).write_text('{}')
    return path


def test_weight_index_alone_is_not_complete(tmp_path):
    p=cache(tmp_path);(p/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'layer':'part1.safetensors'}}))
    assert not m.snapshot_complete(p)
    (p/'part1.safetensors').write_bytes(b'fake weights')
    assert m.snapshot_complete(p)


def test_empty_and_broken_shards_rejected(tmp_path):
    p=cache(tmp_path);(p/'model.safetensors').touch();assert not m.snapshot_complete(p)
    (p/'model.safetensors').unlink();(p/'model.safetensors').symlink_to(tmp_path/'missing')
    assert not m.snapshot_complete(p)


def test_insufficient_disk_fails_before_download():
    with pytest.raises(RuntimeError,match='No download was started'):
        m.check_space([{'estimated_missing_bytes':32_500_000_000}],11_000_000_000)
    assert m.check_space([{'estimated_missing_bytes':0}],100)==0


def test_missing_model_download_uses_frozen_revision(tmp_path,monkeypatch):
    import huggingface_hub
    import types
    directory=cache(tmp_path/'snapshot');(directory/'model.safetensors').write_bytes(b'weights')
    missing=dict(reader='Qwen',model='Qwen/Qwen3-8B',revision='pinned-sha',complete=False,estimated_missing_bytes=16_400_000_000)
    calls=[]
    monkeypatch.setattr(m,'inspect_panel',lambda cfg:[missing] if not calls else [dict(missing,complete=True)])
    monkeypatch.setattr(m,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=100_000_000_000))
    def download(model,**kwargs):calls.append((model,kwargs));return str(directory)
    monkeypatch.setattr(huggingface_hub,'snapshot_download',download)
    assert m.ensure_panel({},download=True)[0]['complete']
    assert calls==[('Qwen/Qwen3-8B',{'revision':'pinned-sha'})]


def test_space_failure_does_not_start_download(tmp_path,monkeypatch):
    import huggingface_hub
    import types
    missing=dict(reader='Qwen',model='Qwen/Qwen3-8B',revision='sha',complete=False,estimated_missing_bytes=16_400_000_000)
    monkeypatch.setattr(m,'inspect_panel',lambda cfg:[missing]);monkeypatch.setattr(m,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=11_000_000_000))
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:pytest.fail('Download must not start'))
    with pytest.raises(RuntimeError,match='No download was started'):m.ensure_panel({},download=True)


def test_complete_cache_skips_download(tmp_path,monkeypatch):
    import huggingface_hub
    import types
    ready=dict(reader='Mistral',model='model',revision='sha',complete=True,estimated_missing_bytes=0)
    monkeypatch.setattr(m,'inspect_panel',lambda cfg:[ready]);monkeypatch.setattr(m,'cache_root',lambda:tmp_path)
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:types.SimpleNamespace(free=100))
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:pytest.fail('Completed snapshot must not redownload'))
    assert m.ensure_panel({},download=True)==[ready]


def test_smoke_uses_longest_evidence_without_quality_selection(tmp_path):
    from src.week10.root_cause.storage import Store,atomic
    store=Store(tmp_path/'runs/C-Mistral',{'synthetic':True})
    for q,length,quality in [('a',10,1.),('b',100,0.)]:
        for action in ('full','qcs256','no_evidence'):
            store.put([q,action],dict(question_id=q,action=action,evidence_tokens_canonical=length,
                evidence_text=q+action,f1=quality))
    atomic(store.path/'complete.json',{'done':True})
    q,evidence=m.smoke_inputs(tmp_path)
    assert q=='b' and evidence['full']=='bfull'


def test_cached_qualification_skips_gpu_loading(tmp_path,monkeypatch):
    import huggingface_hub
    from src.week10.root_cause.storage import atomic,digest,file_hash
    directory=cache(tmp_path/'model');(directory/'model.safetensors').write_bytes(b'fake')
    dataset=tmp_path/'dev.jsonl';dataset.write_text(json.dumps(dict(id='q',question='Question'))+'\n')
    cfg=dict(output_root=str(tmp_path/'results'),dataset=str(dataset),models={'Mistral':dict(id='model',revision='sha')})
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(directory))
    evidence={'full':'Exact source','qcs256':'Exact','no_evidence':''}
    monkeypatch.setattr(m,'smoke_inputs',lambda root:('q',evidence))
    identity=dict(reader=cfg['models']['Mistral'],checkpoint_signature=m.weight_signature(directory),
        evidence_hash=digest(evidence),question_hash=digest('Question'),
        runtime_sha256=file_hash(m.ROOT/'src/week10/root_cause/runtime.py'),
        setup_sha256=file_hash(SCRIPT),precision='fp16',attention='efficient_sdpa',max_new_tokens=128)
    atomic(Path(cfg['output_root'])/'reader_setup/Mistral.json',dict(passed=True,identity=identity))
    import src.week10.root_cause.runtime as runtime
    monkeypatch.setattr(runtime,'TextReader',lambda *a:pytest.fail('Cached qualification must not load a reader'))
    m.qualify_one(cfg,'Mistral')

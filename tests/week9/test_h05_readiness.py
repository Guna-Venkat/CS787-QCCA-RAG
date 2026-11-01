"""H0.5 checks use only code, historical inspected splits and fabricated fixtures."""
import copy
import json
from pathlib import Path
import subprocess
import pytest
from src.week9.final_h import protocol as p
from src.week9.final_h05 import readiness as h,history,producer,runner
from src.week9.final_h05.storage import Shards,membership,pair_membership,atomic_json
from src.week9.final_h05.materialize import TEST_FORMATTED,TEST_RAW,claim_required


@pytest.fixture(autouse=True)
def deny_real_test_reads(monkeypatch):
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.absolute() in (TEST_FORMATTED.absolute(),TEST_RAW.absolute()):
            pytest.fail('Real TEST input opened by readiness test')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)


@pytest.fixture(scope='module')
def registry():return h.make_registry(h.sources())


@pytest.fixture(scope='module')
def fabricated(tmp_path_factory):
    from src.week9.final_h05.synthetic import run
    root=tmp_path_factory.mktemp('h05_fabricated_only')
    return root,run(root)


def test_branch():assert p.branch_guard()=='gagan'


def test_v1_byte_identical():assert p.file_hash(h.V1/'w9h_pretest_freeze_manifest.json')==h.V1_SHA


def test_exact_recovery_class():assert history.recovery()['classification']=='RECOVERED_EXACT'


def test_reject_unverified_w8(monkeypatch):
    old=p.read
    monkeypatch.setattr(p,'read',lambda path:dict(old(path),classification='NOT_RECOVERED') if Path(path).name=='recovery.json' else old(path))
    with pytest.raises(ValueError,match='substitute'):
        history.recovery()


def test_all_twelve_ids(registry):assert len(registry['systems'])==12 and set(registry['systems'])==set(p.REQUIRED)


def test_commands_deterministic(registry):
    other=h.make_registry(next(iter(registry['systems'].values()))['source_hashes'])
    assert registry==other


def test_every_producer_pins_sources(registry):
    for entry in registry['systems'].values():
        assert 'scripts/week9/h05/produce.py' in entry['source_hashes']
        assert 'src/week9/final_h05/adapters.py' in entry['source_hashes']
        assert 'src/week9/utilization_f/prompts.py' in entry['source_hashes']


@pytest.mark.parametrize('system',producer.ORDER)
def test_dry_run_no_test_or_generation(tmp_path,registry,system,monkeypatch):
    atomic_json(tmp_path/'producer_registry.json',registry)
    monkeypatch.setattr(producer,'ensure_shared',lambda *args:pytest.fail('Dry-run invoked TEST materialization'))
    result=producer.dry_run(tmp_path,system)
    assert result['actual_lm_calls']==0 and not result['test_records_opened']


def test_fixture_all_twelve(fabricated):
    _,r=fabricated
    assert r['all_adapters_exercised'] and len(r['systems'])==12
    assert r['real_lm_generation_calls']==0 and r['real_encoder_calls']==0


def test_fixture_no_test_content(fabricated):
    _,r=fabricated
    assert all(q.startswith('SYNTHETIC-') for q in r['fixture_membership'])
    assert r['test_records_opened'] is False


def test_fixture_gate_branches(fabricated):
    _,r=fabricated
    assert set(r['w8_branches'])=={'STOP','PROMOTE'} and set(r['g2_branches'])=={False,True}


def ids(q='SYNTHETIC-q',paper='SYNTHETIC-p',split='test'):
    return [dict(question_id=q,paper_id=paper,split=split)]


def test_common_membership():
    pair_membership(ids(),ids())
    with pytest.raises(ValueError,match='differs'):pair_membership(ids(),ids(paper='other'))


def test_duplicate_membership():
    with pytest.raises(ValueError,match='Duplicate'):membership(ids()+ids())


def test_split_contamination():
    with pytest.raises(ValueError,match='contamination'):membership(ids(split='dev'))


def test_explicit_h2_gate_first(tmp_path):
    with pytest.raises(ValueError,match='disabled'):runner.confirm(tmp_path)


def test_h2_missing_system(tmp_path,registry,monkeypatch):
    r=copy.deepcopy(registry);r['systems'].pop('D0');atomic_json(tmp_path/'producer_registry.json',r)
    monkeypatch.setattr(runner,'validate_v2',lambda path:{})
    with pytest.raises(ValueError,match='12 producers'):runner.confirm(tmp_path,True)


@pytest.fixture
def small_freeze(tmp_path,monkeypatch):
    dep=tmp_path/'dependency';dep.write_text('pinned')
    m=dict(schema='W9-H-pretest-v2',method=p.method_definition(),success_criteria=p.success_criteria(),
        dependency_hashes={str(dep):p.file_hash(dep)},h0_passed=True,blockers=[],h05_readiness=dict(passed=True),
        w8_recovery=dict(classification='RECOVERED_EXACT'),producer_contracts=dict.fromkeys(p.REQUIRED),producer_runtime=h.runtime())
    m['manifest_digest']=p.digest(m);p.write_once(tmp_path/'w9h_pretest_freeze_manifest.json',m)
    monkeypatch.setattr(h,'V2',tmp_path)
    return tmp_path,m,dep


def test_h2_changed_hash(small_freeze):
    root,m,dep=small_freeze;dep.write_text('changed')
    with pytest.raises(ValueError,match='dependency changed'):h.validate_v2(root)


def test_h2_blocked_h0(small_freeze):
    root,m,_=small_freeze;path=root/'w9h_pretest_freeze_manifest.json';path.chmod(0o644);path.unlink()
    m.pop('manifest_digest');m.update(h0_passed=False,blockers=['missing producer']);m['manifest_digest']=p.digest(m);p.write_once(path,m)
    with pytest.raises(ValueError,match='H0 BLOCKED'):h.validate_v2(root)


def test_h2_pretest_claim_required(tmp_path):
    with pytest.raises(ValueError,match='explicit matching H2'):claim_required(tmp_path,dict(manifest_digest='frozen'))


def test_frozen_g2_threshold():assert p.read(p.G_ROOT/'final_fit/threshold.json')['threshold']==.70


def test_frozen_g2_features():assert len(p.read(p.G_ROOT/'final_fit/feature_schema.json')['feature_columns'])==32


def test_e3_k():assert p.method_definition()['sentence_k']==3


def test_primary_unchanged():assert p.method_definition()['primary']=='E3-k3 + F0'


def test_f2_prompt_hash_unchanged():assert p.method_definition()['prompts']['F2-A']==p.read(p.F_ROOT/'prompts.json')['F2-A']


def test_w8_archive_hashes():
    r=history.recovery()
    for original,item in r['files'].items():
        assert p.file_hash(p.ROOT/item['archive_path'])==item['sha256']
        blob=subprocess.check_output(['git','rev-parse',r['commit']+':'+original],cwd=p.ROOT,text=True).strip()
        assert blob==item['git_blob']


def test_resume_rejects_config_mismatch(tmp_path):
    Shards(tmp_path,dict(freeze='a',config='original')).put('SYNTHETIC-q',dict(value=1))
    with pytest.raises(ValueError,match='Resume identity'):Shards(tmp_path,dict(freeze='a',config='modified'))


def test_shard_roundtrip_and_no_replacement(tmp_path):
    rows=Shards(tmp_path,dict(freeze='same'));rows.put('SYNTHETIC-q',dict(value=1))
    assert rows.get('SYNTHETIC-q')==dict(value=1)
    with pytest.raises(ValueError,match='overwritten'):rows.put('SYNTHETIC-q',dict(value=2))


def test_shard_checksum_rejects_tampering(tmp_path):
    store=Shards(tmp_path,dict(freeze='same'));store.put('SYNTHETIC-q',dict(value=1))
    row=p.read(store.path('SYNTHETIC-q'));row['payload']['value']=2;atomic_json(store.path('SYNTHETIC-q'),row)
    with pytest.raises(ValueError,match='checksum'):store.get('SYNTHETIC-q')


def test_versioned_paths(registry):
    for r in registry['systems'].values():
        assert r['result_path'].startswith('results/week9/w9h_final/v2/')
        assert r['receipt_path'].startswith('results/week9/w9h_final/v2/')


def test_original_sources_unchanged():
    old=p.read(h.V1/'w9h_pretest_freeze_manifest.json')
    p.verify_hashes(old['relevant_working_tree_sources'])


def test_matrix_all_twelve():
    rows=h.compatibility_matrix()
    assert {r['system'] for r in rows}==set(p.REQUIRED) and len(rows)==12
    assert all(r['executable'] and not r['latency_comparable'] for r in rows)


def test_exact_success_criteria_unchanged():assert p.success_criteria()==p.read(h.V1/'w9h_pretest_freeze_manifest.json')['success_criteria']


def test_historical_metadata_not_new_methods(registry):
    assert registry['systems']['BM25_k8']['method_config']['decoding']['max_new_tokens']==64
    assert registry['systems']['D0']['method_config']['generator']['dtype']=='torch.bfloat16'
    assert registry['systems'][p.PRIMARY]['method_config']['generator']['dtype']=='torch.float16'


def test_json_order_does_not_determine_execution(registry,tmp_path):
    p.write_once(tmp_path/'registry.json',registry)
    saved=p.read(tmp_path/'registry.json')
    assert saved['order'][0]=='D0' and saved['order'][-1]=='original_SARA_G0'


def test_new_notebook_defaults_and_v2():
    notebook=p.read(p.ROOT/'notebooks/week9/week9h_final_evaluation.ipynb')
    text='\n'.join(''.join(c['source']) for c in notebook['cells'])
    assert 'RUN_FINAL_CONFIRMATION = False' in text and 'RUN_ROBUSTNESS = False' in text
    assert 'final_h05.runner' in text and 'w9h_final_v2.json' in text


def test_fixture_g2_costs(fabricated):
    root,_=fabricated;rows=p.read(root/'sealed_predictions.json')
    f0={r['question_id']:r for r in rows if r['system']==p.PRIMARY}
    for r in rows:
        if r['system']==p.VERIFY:
            assert r['total_query_prompt_tokens']==f0[r['question_id']]['total_query_prompt_tokens']
            assert r['num_lm_calls']==2


def test_synthetic_h2_receipts_order_seal_and_one_shot(tmp_path,fabricated,registry,monkeypatch):
    """Exercise the real coordinator with fabricated producer outputs; no subprocess or TEST access."""
    from src.week9.final_h05.materialize import dump_lines
    fixture_root,_=fabricated;all_rows=p.read(fixture_root/'sealed_predictions.json')
    freeze=dict(manifest_digest='SYNTHETIC_COORDINATOR_ONLY',model=p.read(p.F_ROOT/'frozen_w9f_intervention.json')['generator_identity'],method=p.method_definition())
    r=copy.deepcopy(registry);s=h.spec(r)
    for name in r['order']:
        r['systems'][name]['result_path']=str(tmp_path/'predictions'/f'{name}.jsonl')
        r['systems'][name]['receipt_path']=str(tmp_path/'receipts'/f'{name}.json')
        s['systems'][name].update(predictions_path=r['systems'][name]['result_path'],receipt_path=r['systems'][name]['receipt_path'])
    s['membership_path']=str(tmp_path/'shared/membership.jsonl');s['labels_path']=str(tmp_path/'shared/labels.jsonl')
    atomic_json(tmp_path/'producer_registry.json',r);atomic_json(tmp_path/'test_materialization_spec.json',s)
    monkeypatch.setattr(runner,'validate_v2',lambda root:freeze)
    executed=[]
    def fake_run(command,**kwargs):
        name=command[command.index('--system')+1];executed.append(name)
        assert p.read(tmp_path/'h2_claim.json')['freeze_digest']==freeze['manifest_digest']
        records=[row for row in all_rows if row['system']==name]
        if name=='D0':
            dump_lines(s['membership_path'],[dict(question_id=row['question_id'],paper_id=row['paper_id'],split='test') for row in records])
            dump_lines(s['labels_path'],[dict(question_id=row['question_id'],paper_id=row['paper_id'],evaluation=dict(answers=['synthetic answer'],answers_reformatted=['synthetic answer'])) for row in records])
        entry=s['systems'][name];dump_lines(entry['predictions_path'],records)
        p.write_once(entry['receipt_path'],dict(complete=True,system=name,split='test',freeze_digest=freeze['manifest_digest'],
            method_config=entry['method_config'],producer_source_hashes=entry['producer_source_hashes'],predictions_sha256=p.file_hash(entry['predictions_path'])))
    monkeypatch.setattr(runner.subprocess,'run',fake_run)
    runner.confirm(tmp_path,allow=True)
    assert executed==producer.ORDER
    sealed=p.read(tmp_path/'h2_seal.json')
    assert sealed['complete'] and sealed['queries']==2 and sealed['systems']==producer.ORDER
    p.verify_hashes(sealed['artifact_hashes'])
    with pytest.raises(ValueError,match='already sealed'):runner.confirm(tmp_path,allow=True,resume=True)

"""Presentation defaults and deterministic archive restore; no model inference."""
from pathlib import Path
import hashlib,json,subprocess
import pytest
from streamlit.testing.v1 import AppTest
from demo.curated.ui import available_methods
from demo.live.preset import load_preset,passage_status
from demo.live.protocol import CONTROLLED,NATIVE
from demo.live.engine import run
from demo.tests.test_live import FakeResources


def host(tmp_path,technical=False):
 script=tmp_path/'freeze_host.py'
 script.write_text('''import streamlit as st
from demo.live import ui
from demo.tests.test_live import FakeResources
ui.resources=lambda:FakeResources()
ui.available=lambda:dict(sfr_local=True,qwen_local=True,gpu_free_mib=None)
ui.render(technical='''+repr(technical)+''')
''')
 return AppTest.from_file(str(script)).run(timeout=20)

def test_presentation_is_controlled_only(tmp_path):
 app=host(tmp_path)
 control=next(s for s in app.selectbox if s.label=='Comparison protocol')
 assert control.value==CONTROLLED
 assert control.options==['Controlled retriever-only comparison']
 assert 'Pipeline-native comparison' not in control.options
 assert not app.exception

def test_researcher_retains_native_variant(tmp_path):
 app=host(tmp_path,True)
 control=next(s for s in app.selectbox if s.label=='Comparison protocol')
 assert control.value==CONTROLLED and 'Pipeline-native comparison' in control.options
 control.set_value(NATIVE).run()
 assert not app.exception
 assert any('not retrieval alone' in c.value or 'not a pure' in c.value for c in app.caption)

def test_historical_lexical_qbgf_is_researcher_only():
 assert 'QBGF_FULL' not in available_methods('BM25',False)
 assert 'QBGF_FULL' in available_methods('BM25',True)
 assert 'QBGF_FULL' in available_methods('SFR',False)

def test_preset_loads_without_generating(tmp_path):
 app=host(tmp_path)
 next(b for b in app.button if b.label=='Budget-sensitive multi-aspect question').click().run()
 assert not app.exception
 assert len([a for a in app.text_area if a.label.startswith('Passage ')])==15
 assert next(a for a in app.text_area if a.label=='Your query').value==load_preset()['query']
 assert not app.session_state['live_results']

@pytest.mark.parametrize('method',['RAW_SKIP','FIXED_SENTENCE','QBGF_FULL'])
def test_p7_p9_states_are_actual_allocation(method):
 p=load_preset();r=run(p['query'],p['passages'],method,1024,'BM25',FakeResources(),True,generate_answer=False,scoring_mode=CONTROLLED)
 rows=passage_status(r)
 assert {x['Passage'] for x in rows}=={'P7','P9'}
 for row in rows:
  chosen=[u for u in r['selection']['units'] if u['document_id']==row['Passage']]
  assert row['Selected_units']==len(chosen)
  assert row['Allocation'].startswith('SELECTED')==bool(chosen)

def test_archive_restore_idempotent_and_fail_closed(tmp_path):
 from demo.restore_curated_bundle import restore
 dest=tmp_path/'restored'
 receipt=restore(dest);assert receipt['verified_files']==3
 paths=list(dest.iterdir());mtimes={p.name:p.stat().st_mtime_ns for p in paths}
 assert restore(dest)==receipt
 assert {p.name:p.stat().st_mtime_ns for p in paths}==mtimes
 (dest/'manifest.json').write_text('{}')
 with pytest.raises(ValueError,match='refusing to overwrite'):restore(dest)

def test_archive_tamper_rejected(tmp_path):
 from demo.restore_curated_bundle import restore
 root=Path(__file__).resolve().parents[2];src=root/'demo/releases/curated-demo-v1.tar.gz'
 bad=tmp_path/'bad.tar.gz';data=bytearray(src.read_bytes());data[-1]^=1;bad.write_bytes(data)
 with pytest.raises(ValueError,match='SHA256 mismatch'):restore(tmp_path/'out',archive=bad)

def test_only_selective_ignores():
 root=Path(__file__).resolve().parents[2]
 def ignored(name):return subprocess.run(['git','check-ignore','--no-index',name],cwd=root,capture_output=True).returncode==0
 assert not ignored('demo/app.py')
 assert not ignored('demo/assets/live_tokenizer/tokenizer.json')
 assert not ignored('demo/releases/curated-demo-v1.tar.gz')
 assert ignored('demo/.venv/bin/python') and ignored('demo/runtime/custom.json')
 assert ignored('demo/replay_data/v1/bundle.json')
 assert not ignored('demo/replay_data/v1/PUBLISHED.json')

def test_compressed_primary_source_is_read_only(tmp_path,monkeypatch):
 import gzip
 import demo.data_loader as loader
 data=b'{"frozen":true}\n';path=tmp_path/'results/protocol.json'
 path.parent.mkdir();path.with_suffix('.json.gz').write_bytes(gzip.compress(data))
 monkeypatch.setattr(loader,'ROOT',tmp_path)
 monkeypatch.setattr(loader,'SOURCES',{'fixture':'results/protocol.json'})
 assert loader.load('fixture')=={'frozen':True}
 assert loader.sha(path)==hashlib.sha256(data).hexdigest()
 assert not path.exists() # No uncompressed research artifact is created.
 (tmp_path/'demo').mkdir()
 (tmp_path/'demo/artifact_manifest.json').write_text(json.dumps({'sources':{'fixture':{'path':'results/protocol.json','sha256':hashlib.sha256(data).hexdigest()}}}))
 assert loader.verify_manifest()

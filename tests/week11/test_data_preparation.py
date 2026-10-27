"""Offline checks for dataset preparation; no model calls or network requests."""
import json
import urllib.error
import pytest
from scripts.week11 import prepare_datasets as prep


def test_hotpot_layout_conversion_preserves_strings_and_order():
    row=dict(id='q',question=' Why? ',answer='é',type='bridge',level='hard',
        context=dict(title=['B','A'],sentences=[['  First.\n','Second!'],['Third.']]),
        supporting_facts=dict(title=['A','B'],sent_id=[0,1]))
    result=prep.original_hotpot_rows([row])[0]
    assert result['_id']=='q' and result['question']==' Why? ' and result['answer']=='é'
    assert result['context']==[['B',['  First.\n','Second!']],['A',['Third.']]]
    assert result['supporting_facts']==[['A',0],['B',1]]


def test_hotpot_conversion_refuses_truncated_columns():
    row=dict(id='q',question='q',answer='a',type='bridge',level='hard',
        context=dict(title=['A'],sentences=[]),supporting_facts=dict(title=[],sent_id=[]))
    with pytest.raises(ValueError):prep.original_hotpot_rows([row])


def test_existing_download_is_not_replaced(tmp_path,monkeypatch):
    path=tmp_path/'existing';path.write_bytes(b'locked')
    monkeypatch.setattr(prep.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Unexpected network request'))
    prep.download('https://unused.example',path)
    assert path.read_bytes()==b'locked'


def test_dns_failure_uses_pinned_hotpot_source(tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    called=[]
    def download(url,target):
        called.append(url)
        if url==prep.SOURCES['HotpotQA']:raise urllib.error.URLError('DNS unavailable')
        target.write_bytes(b'fabricated parquet fixture')
    class FixtureTable:
        def to_pylist(self):
            return [dict(id=str(i),question='q',answer='a',type='bridge',level='hard',
                context=dict(title=['A'],sentences=[['Exact source.']]),
                supporting_facts=dict(title=['A'],sent_id=[0])) for i in range(7405)]
    monkeypatch.setattr(prep,'download',download)
    monkeypatch.setattr(pq,'read_table',lambda p:FixtureTable())
    path=tmp_path/'hotpot.json';prep.prepare_hotpot(path)
    assert called==[prep.SOURCES['HotpotQA'],prep.HOTPOT_FALLBACK]
    receipt=json.loads((tmp_path/'hotpot_source.json').read_text())
    assert receipt['revision']==prep.HOTPOT_REVISION and receipt['split']=='validation'
    assert receipt['converted_sha256']==prep.file_hash(path)
    assert len(json.loads(path.read_text()))==7405


def test_incomplete_fallback_is_not_published(tmp_path,monkeypatch):
    import pyarrow.parquet as pq
    def download(url,target):
        if url==prep.SOURCES['HotpotQA']:raise urllib.error.URLError('DNS unavailable')
        target.write_bytes(b'fixture')
    class EmptyTable:
        def to_pylist(self):return []
    monkeypatch.setattr(prep,'download',download)
    monkeypatch.setattr(pq,'read_table',lambda p:EmptyTable())
    path=tmp_path/'hotpot.json'
    with pytest.raises(ValueError,match='7405'):prep.prepare_hotpot(path)
    assert not path.exists()

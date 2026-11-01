"""Lossless storage and retrieval-equivalence regressions; no full experiments."""
import json
import sqlite3
import numpy as np
import pytest
from src.week9.retrieval_cache import RetrievalCache, read_rows
from src.week9.retrieval import rank_candidates, bm25, unit_vectors
from src.week9.evidence_alignment import alignment, retrieval_metrics
from src.week9.caching import read_jsonl
from tests.week9.test_week9_suite import Encoder, chunks


def test_compressed_cache_resume_filter_checksum_and_rollback(tmp_path):
    path=tmp_path/'rows.sqlite3';identity={'source':'one'}
    with RetrievalCache(path,identity) as cache:
        for q in ['q1','q2']:
            row=dict(question_id=q,paper_id='p',representation='ORIG',retriever='BM25',retrieval_width=10,candidates=chunks())
            key=cache.key(q,'p',[],'op');cache.put(key,row)
            assert cache.rows[key]['candidates']==row['candidates']
        assert cache.rows.count(question_id='q1')==1
        assert len(list(cache.rows.query(question_id='q2')))==1
        assert all('candidates' not in r for r in cache.metric_rows())
    assert len(read_jsonl(path))==2
    assert [r['question_id'] for r in read_rows(path,{'question_id':'q2'})]==['q2']
    with RetrievalCache(path,identity,True) as cache:assert len(cache.rows)==2
    with pytest.raises(ValueError):RetrievalCache(path,{'source':'changed'},True)
    with pytest.raises(FileExistsError):RetrievalCache(path,identity)
    with pytest.raises(ValueError):list(read_rows(path,{'unsafe':'x'}))
    # An uncommitted transaction is rolled back on connection loss.
    db=sqlite3.connect(path);db.execute("DELETE FROM entries WHERE question_id='q1'");db.close()
    assert len(list(read_rows(path)))==2
    db=sqlite3.connect(path);db.execute("UPDATE entries SET payload=? WHERE question_id='q1'",(b'broken',));db.commit();db.close()
    with pytest.raises(Exception):list(read_rows(path))


def test_shared_scores_and_alignment_are_exactly_equivalent():
    enc=Encoder();cs=chunks(6);evidence=['apple fact 2','apple fact 4'];query='apple zebra'
    vectors=unit_vectors(enc.encode([c['text'] for c in cs]));qvec=enc.encode([query])[0]
    sparse=bm25(query,[c['text'] for c in cs]);dense=unit_vectors(vectors) @ unit_vectors([qvec])[0]
    aligned={c['chunk_id']:alignment(c,evidence) for c in cs}
    for method in ['BM25','SEMANTIC','RRF']:
        full=rank_candidates(query,'p',cs,enc,method,6,vectors,sparse_scores=sparse,dense_scores=dense)
        for width in [1,3,6,10]:
            old=rank_candidates(query,'p',cs,enc,method,width,vectors)
            assert old==full[:width]
            assert retrieval_metrics(old,cs,evidence)==retrieval_metrics(full[:width],cs,evidence,alignments=aligned)


def test_a_small_grid_resume_reuses_all_completed_work(tmp_path, monkeypatch, capsys):
    from src.week9 import runner
    from tests.week9.test_week9_suite import document
    class CountingEncoder(Encoder):
        query_calls = 0
        def encode(self, texts):
            if texts == ['apple zebra?']:
                self.query_calls += 1
            return super().encode(texts)
    enc=CountingEncoder()
    monkeypatch.setattr(runner,'source_document',lambda *args:document())
    monkeypatch.setattr(runner,'original_chunks',lambda *args:chunks())
    monkeypatch.setattr(runner,'semantic_chunks',lambda *args,**kwargs:chunks())
    monkeypatch.setattr(runner,'fixed_chunks',lambda *args:chunks())
    monkeypatch.setattr(runner,'evaluation_evidence',lambda *args:['apple fact 2'])
    records=[dict(inference=dict(question_id=q,paper_id='p',question='apple zebra?'),original_passages=[document()['text']]) for q in ['q1','q2']]
    cfg=dict(semantic_grid=[{}],fixed_overlap=0,retrieval_widths=[1,3])
    identity={'test':'grid'}
    with RetrievalCache(tmp_path/'results.sqlite3',identity) as results, RetrievalCache(tmp_path/'chunks.sqlite3',identity) as stored, RetrievalCache(tmp_path/'embeddings.sqlite3',identity) as embeddings:
        runner.run_a(cfg,records,{'raw':'unused'},results,tmp_path,enc,{},embeddings,stored)
        assert len(results.rows)==36 and enc.query_calls==2
        before=list(results.rows.values())
        runner.run_a(cfg,records,{'raw':'unused'},results,tmp_path,enc,{},embeddings,stored)
        assert enc.query_calls==2 and list(results.rows.values())==before
        assert 'already complete' in capsys.readouterr().out


def test_cache_crash_recovers_durable_batches_and_rejects_second_writer(tmp_path):
    import subprocess
    import sys
    path=tmp_path/'crash.sqlite3'
    program='''
import os,sys
from src.week9.retrieval_cache import RetrievalCache
cache=RetrievalCache(sys.argv[1], {'run':'crash'})
for i in range(129):
    key=cache.key(str(i),'p',[],'operation')
    cache.put(key, {'question_id':str(i),'paper_id':'p','value':i})
os._exit(0)
'''
    subprocess.run([sys.executable,'-c',program,str(path)],check=True)
    with RetrievalCache(path,{'run':'crash'},True) as cache:
        assert len(cache.rows)==128
        with pytest.raises(BlockingIOError):
            RetrievalCache(path,{'run':'crash'},True)
        key=cache.key('128','p',[],'operation')
        cache.put(key,{'question_id':'128','paper_id':'p','value':128})
    assert len(list(read_rows(path)))==129

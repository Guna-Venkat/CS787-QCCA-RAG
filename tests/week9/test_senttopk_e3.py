"""Tiny E3 fixtures: source identity, semantic rank, oracles and safe notebook."""
import json,re,copy
from pathlib import Path
import numpy as np
import pytest
from src.week9.senttopk_e3.sentences import spans,inventory,validate,rank,select,evidence,diagnostics
from src.week9.senttopk_e3.embeddings import cached,embedding_key,encoder_identity
from src.week9.senttopk_e3.protocol import GRID,validate_old_e_row
from src.week9.senttopk_e3.analysis import oracle,choose_fixed,dynamic_decision,analyze
from src.week9.caching import RowCache,atomic_json
from src.week9.retrieval_cache import RetrievalCache
from src.week9.provenance import digest,file_hash

class Tokenizer:
    def encode(self,text,**kwargs):return re.findall(r'\w+|[^\w\s]',text)

@pytest.fixture
def chunks():return [dict(chunk_id='later',paper_id='p',text='Third sentence. Fourth sentence.',char_start=100,char_end=132),dict(chunk_id='earlier',paper_id='p',text='First sentence. Second sentence.',char_start=0,char_end=32)]


def test_academic_sentence_split_deterministic():
    text='Dr. Smith et al. use Fig. 2, e.g. a score of 3.14. Next result!\n\nA. Jones cites Eq. 4. Final statement.'
    a=spans(text);assert a==spans(text)
    assert [text[s:e] for s,e in a]==['Dr. Smith et al. use Fig. 2, e.g. a score of 3.14.','Next result!','A. Jones cites Eq. 4.','Final statement.']
    assert all(text[s:e].strip()==text[s:e] for s,e in a)


def test_exact_sentence_source_offsets_and_forgery(chunks):
    s=inventory('q','p',chunks,Tokenizer());validate(s,chunks,'p')
    assert len(s)==4 and s==inventory('q','p',chunks,Tokenizer())
    assert s[0]['original_chunk_rank']==1 and s[2]['original_chunk_rank']==0
    for item in s:
        chunk=next(c for c in chunks if c['chunk_id']==item['chunk_id'])
        assert chunk['text'][item['char_start']:item['char_end']]==item['text']
    bad=copy.deepcopy(s);bad[0]['text']='invented'
    with pytest.raises(ValueError,match='identity'):validate(bad,chunks,'p')
    assert spans(' \n\t ')==[]


def test_cosine_rank_ties_topk_and_source_presentation(chunks):
    s=inventory('q','p',chunks,Tokenizer())
    vectors=np.array([[1,0],[0,1],[1,0],[-1,0]])
    ranked=rank('question',s,[1,0],vectors)
    # Ties prefer original document order, not incoming DraftSemantic chunk order.
    assert [r['text'] for r in ranked[:2]]==['First sentence.','Third sentence.']
    selected=select(ranked,3)
    assert [r['text'] for r in selected]==['First sentence.','Third sentence.','Fourth sentence.']
    assert [r['selected_rank'] for r in selected]==[1,2,3]
    assert [r['presentation_rank'] for r in selected]==[1,2,3]
    # Make semantic selection reverse source order; presentation must restore it.
    ranked=rank('question',s,[1,0],[[.9,.1],[0,1],[1,0],[-1,0]])
    picked=select(ranked,2)
    assert picked[0]['document_char_start']<picked[1]['document_char_start']
    assert len(select(ranked,8))==4 and len({r['sentence_id'] for r in select(ranked,8)})==4
    assert evidence(picked)=='\n\n'.join(r['text'] for r in picked)


def test_selection_vs_presentation_ranks_are_distinct(chunks):
    s=inventory('q','p',chunks,Tokenizer())
    ranked=rank('question',s,[1,0],[[1,0],[.9,.1],[.8,.2],[0,1]])
    picked=select(ranked,3)
    assert [r['selected_rank'] for r in picked]==[3,1,2]
    assert [r['presentation_rank'] for r in picked]==[1,2,3]


def test_no_gold_or_draft_in_scoring_api(chunks):
    s=inventory('q','p',chunks,Tokenizer())
    with pytest.raises(ValueError):rank({'question':'q','gold':'answer'},s,[1,0],[[1,0]]*4)
    bad=copy.deepcopy(s);bad[0]['gold_label']=True
    with pytest.raises(ValueError):rank('q',bad,[1,0],[[1,0]]*4)
    assert rank('q',[],[1,0],[])==[]


def test_embedding_cache_text_model_normalization_identity(tmp_path):
    meta=dict(model_id='canonical',revision='r1',normalization='unit')
    class Encoder:
        n=0
        def encode(self,texts):self.n+=1;return np.array([[len(t),1.] for t in texts])
    encoder=Encoder()
    with RetrievalCache(tmp_path/'sentences.sqlite3',dict(encoder=meta),True) as cache:
        first,a=cached(cache,['alpha','beta'],'sentence',encoder,meta)
        second,b=cached(cache,['alpha','beta'],'sentence',encoder,meta)
        assert np.array_equal(first,second) and encoder.n==1
        assert b['current_compute_seconds']==0 and b['conceptual_compute_seconds']>0
        key=embedding_key(cache,'alpha','sentence',meta)
        assert key!=embedding_key(cache,'changed','sentence',meta)
        assert key!=embedding_key(cache,'alpha','query',meta)
        assert key!=embedding_key(cache,'alpha','sentence',{**meta,'revision':'r2'})
        assert key!=embedding_key(cache,'alpha','sentence',{**meta,'normalization':'different'})


def test_encoder_revision_must_match_frozen_files(tmp_path,monkeypatch):
    from src.week9.senttopk_e3 import embeddings as e
    snapshot=tmp_path/'fixed-revision';snapshot.mkdir();p=snapshot/'config.json';p.write_text('{}')
    monkeypatch.setattr(e,'snapshot_download',lambda *a,**kw:str(snapshot))
    result=encoder_identity('sentence-transformers/all-MiniLM-L6-v2',{str(p):file_hash(p)})
    assert result['revision']=='fixed-revision'
    p.write_text('{"changed":true}')
    with pytest.raises(ValueError):encoder_identity('sentence-transformers/all-MiniLM-L6-v2',{str(p):'oldhash'})


def test_d_e_chunk_identity_and_hash_preservation(chunks):
    r=dict(paper_id='p',initial_draft='draft',source_selected_chunk_ids=[c['chunk_id'] for c in chunks],selected_chunk_ids=[c['chunk_id'] for c in chunks],source_chunk_hashes={c['chunk_id']:digest(c['text']) for c in chunks})
    validate_old_e_row(r,chunks,'p','draft')
    for changed in [dict(paper_id='other'),dict(source_selected_chunk_ids=['wrong']),dict(source_chunk_hashes={'wrong':'hash'}),dict(initial_draft='changed')]:
        with pytest.raises(ValueError,match='identity'):validate_old_e_row({**r,**changed},chunks,'p','draft')


def fixture_surface():
    rows=[];d0=[]
    for q in ('q1','q2'):
        for k in GRID:
            score=(.5 if k>=2 else .49) if q=='q1' else .4
            rows.append(dict(question_id=q,paper_id='p',policy=f'E3-k{k}',sentence_k=k,f1=score,evidence_tokens=k*5,final_prompt_tokens=20+k,total_query_prompt_tokens=30+k,total_query_generated_tokens=10,num_lm_calls=2))
        d0.append(dict(question_id=q,paper_id='p',policy='D0',sentence_k=0,f1=.495 if q=='q1' else .4,evidence_tokens=0,final_prompt_tokens=5,total_query_prompt_tokens=5,total_query_generated_tokens=3,num_lm_calls=1))
    return rows,d0


def test_intervention_and_full_action_oracles_load_d0_and_tie_smallest():
    rows,d0=fixture_surface()
    forced,fs,_=oracle(rows,d0)
    full,summary,cost=oracle(rows,d0,include_zero=True)
    f=forced.set_index('question_id');a=full.set_index('question_id')
    assert f.loc['q1','exact_oracle_k']==2 and f.loc['q1','epsilon_oracle_k']==1
    assert f.loc['q2','exact_oracle_k']==1
    assert a.loc['q1','epsilon_oracle_k']==0 and a.loc['q2','exact_oracle_k']==0
    assert summary['epsilon']['mean_cumulative_prompt_tokens']==5
    assert not summary['epsilon']['deployable']
    local,_,_=oracle(rows,d0,allowed=[2,3,4])
    assert local.exact_oracle_k.min()==2
    with pytest.raises(ValueError):oracle(rows[:-1],d0)
    with pytest.raises(ValueError):oracle(rows,d0[:1],include_zero=True)


def test_fixed_k_cost_aware_rule():
    a=dict(sentence_k=1,f1=.5,total_query_prompt_tokens=200,final_prompt_tokens=150)
    b=dict(sentence_k=4,f1=.498,total_query_prompt_tokens=100,final_prompt_tokens=80)
    c=dict(sentence_k=8,f1=.496,total_query_prompt_tokens=10,final_prompt_tokens=5)
    assert choose_fixed([a,b,c])==b
    assert choose_fixed([a,{**a,'sentence_k':2,'final_prompt_tokens':149}])['sentence_k']==2


def test_dynamic_is_recommendation_only_and_can_be_no_go():
    c=[dict(sentence_k=k,f1=.5,evidence_tokens=10,final_prompt_tokens=20,total_query_prompt_tokens=30,help=.1,hurt=.1) for k in GRID]
    full=dict(epsilon=dict(f1=.5,mean_evidence_tokens=10,mean_final_prompt_tokens=20,mean_cumulative_prompt_tokens=30))
    forced=dict(epsilon=dict(distribution={str(k):2 if k==1 else 0 for k in GRID}))
    cfg=dict(dynamic_gain=.005,dynamic_cost_saving=.15,quality_tolerance=.003,heterogeneity_min_mass=.15,heterogeneity_help_hurt_range=.05,heterogeneity_min_gain=.003)
    decision=dynamic_decision(full,forced,c,cfg)
    assert decision['recommendation']=='NO-GO' and not decision['controller_trained_or_run']
    full['epsilon']['f1']=.506
    assert dynamic_decision(full,forced,c,cfg)['recommendation']=='GO'


def test_sentence_diversity_is_diagnostic_only(chunks):
    s=inventory('q','p',chunks,Tokenizer());vectors=np.array([[1,0],[1,0],[0,1],[0,1]])
    ranked=rank('q',s,[1,0],vectors);selected=select(ranked,2)
    result=diagnostics(selected,ranked,{a['sentence_id']:v for a,v in zip(s,vectors)})
    assert result['same_chunk_pair_fraction']==1 and result['adjacent_sentence_pair_rate']==1 and result['near_duplicate_sentence_pair_rate']==1
    assert len(selected)==2


def test_old_e_result_artifacts_loaded_without_writes(tmp_path,monkeypatch,chunks):
    from src.week9.senttopk_e3 import protocol as p
    old=tmp_path/'old-e';(old/'protocol').mkdir(parents=True);(old/'protocol/d_input_freeze.json').write_text('{}')
    (old/'fit').mkdir();atomic_json(old/'fit/manifest.json',dict(complete=True,full_fit=True,stage='full',identity_hash=digest({})))
    inf=dict(question_id='q',paper_id='p',initial_draft='draft')
    bundle=dict(records=[dict(inference=inf)],chunks={'q':chunks},seal=dict(model_hashes={},input_hashes={}),baselines=[],paths={},lineage={})
    monkeypatch.setattr(p.old_e,'config',lambda *a:{})
    monkeypatch.setattr(p.old_e,'output_dir',lambda *a:old)
    monkeypatch.setattr(p.old_e,'inputs',lambda *a:bundle)
    monkeypatch.setattr(p.old_e,'identity',lambda *a:{})
    monkeypatch.setattr(p,'encoder_identity',lambda *a:{'revision':'fixture'})
    with RowCache(old/'fit/grounded_e_results.jsonl',dict(stage='full',protocol={})) as cache:
        for policy in ('E0','E1-Q','E1-QD','E2'):
            r=dict(question_id='q',paper_id='p',policy=policy,initial_draft='draft',source_selected_chunk_ids=[c['chunk_id'] for c in chunks],selected_chunk_ids=[c['chunk_id'] for c in chunks],source_chunk_hashes={c['chunk_id']:digest(c['text']) for c in chunks},evidence=dict(evidence_tokens=10))
            cache.put(policy,r)
    files=[f for f in old.rglob('*') if f.is_file()]
    before={str(f):(f.read_bytes(),f.stat().st_mtime_ns) for f in files}
    loaded=p.inputs(dict(old_e_config=str(tmp_path/'config.yaml'),encoder='canonical'))
    assert len(loaded['baselines'])==4
    assert {str(f):(f.read_bytes(),f.stat().st_mtime_ns) for f in files}==before


def test_old_notebook_prefix_preserved_and_all_flags_disabled(monkeypatch):
    import subprocess
    calls=[];monkeypatch.setattr(subprocess,'run',lambda *a,**kw:calls.append(a))
    path=Path('notebooks/week9/week9e_extraction_faithfulness.ipynb')
    n=json.loads(path.read_text());audit=json.loads(Path('results/week9/patch_verification/w9e3_senttopk/notebook_append_audit.json').read_text())
    assert n['cells'][:audit['original_cells']]==audit['prefix_cells']
    ns={}
    for c in n['cells']:
        if c['cell_type']=='code':exec(compile(''.join(c['source']),str(path),'exec'),ns)
    assert not calls and not ns['RUN_E3_FULL'] and not ns['RUN_E3_SMOKE'] and not ns['RUN_FULL']


def test_complete_analysis_freezes_only_new_experiment_and_generates_oracles(tmp_path):
    rows,d0=fixture_surface()
    for r in rows:r.update(delta_f1=0,help=False,hurt=False,answer_changed=True,exact_match=0)
    bases=d0+[{**r,'policy':p} for p in ('D1','D2','D3','E0','E1-Q','E1-QD','E2') for r in d0]
    cfg=dict(quality_tolerance=.003,epsilon=.01,dynamic_gain=.005,dynamic_cost_saving=.15,heterogeneity_min_mass=.15,heterogeneity_help_hurt_range=.05,heterogeneity_min_gain=.003)
    ident=dict(selection_rule='fixture',input_hashes={})
    directory=tmp_path/'new-e3/fit';directory.mkdir(parents=True);(directory/'results.jsonl').write_text('tiny unit fixture')
    result=analyze(rows,bases,cfg,ident,directory,complete=True)
    assert result['selected']['sentence_k']==2
    freeze=json.loads((directory.parent/'frozen_e3_senttopk.json').read_text())
    assert freeze['selected_k']==2 and freeze['diagnostic_only']
    import pandas as pd
    assert len(pd.read_csv(directory/'bootstrap.csv'))==15
    assert set(json.loads((directory/'oracle_summary.json').read_text()))=={'intervention_only','full_action','local_window_intervention_only'}
    assert not json.loads((directory/'dynamic_go_no_go.json').read_text())['controller_trained_or_run']


def test_no_sentence_fallback_does_not_call_any_lm(monkeypatch):
    from src.week9.senttopk_e3 import runner
    from types import SimpleNamespace
    monkeypatch.setattr(runner,'canonical_context',lambda *a:('canonical [/INST]',None,3))
    monkeypatch.setattr(runner,'annotate_outcome',lambda row,*a:row)
    class Backend:
        tokenizer=Tokenizer();tokenizer.name_or_path='fixture';device='cpu';model=SimpleNamespace(dtype='float16')
        def identity(self):return dict(model_id='fixture')
        def generate(self,*a,**kw):raise AssertionError('No-sentence path called the LM')
    inf=dict(question_id='q',paper_id='p',question='Which metric?',initial_draft='original')
    record=dict(inference=inf,canonical={},evaluation={},_d1=dict(retrieval_query_time=.1,selection_latency=.2))
    bundle=dict(lineage={'q':{}},e_lineage={'q':{}},encoder={})
    row=runner.generate(record,[],[],{}, {},1,Backend(),None,None,bundle)
    assert row['prediction']=='original' and row['fail_closed'] and row['failure_reason']=='no_valid_sentences'
    assert not row['answer_generation_called'] and row['num_lm_calls']==1 and row['extraction_generated_tokens']==0

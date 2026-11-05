import ast
from copy import deepcopy
import json
from pathlib import Path
import pytest
from reranker_baseline import protocol as p
from reranker_baseline.minilm_reranker import rank_scores,pack
from reranker_baseline import runner
from reranker_baseline.analysis import comparison_table

class Tok:
    def encode(self,text,**kwargs):return list(map(ord,text))

def units():
    return [dict(unit_id='a',passage_id='p',document_id='d',unit_type='SENTENCE',source_text='A.',
        start_char=0,end_char=2,document_order=[0,0,0,0],token_count=2),
        dict(unit_id='b',passage_id='p',document_id='d',unit_type='SENTENCE',source_text='B.',
        start_char=3,end_char=5,document_order=[0,0,1,0],token_count=2)]
def scores(ss,vals):
    return [dict(sentence_id=s['unit_id'],passage_rank=1,source_position=s['start_char'],sentence_text=s['source_text'],cross_encoder_score=v) for s,v in zip(ss,vals)]

def test_score_order_and_all_tie_breaks():
    rows=[dict(sentence_id=s,passage_rank=p,source_position=x,cross_encoder_score=v) for s,p,x,v in
        [('z',2,0,1),('b',1,2,1),('a',1,2,1),('c',1,0,1),('high',10,100,2)]]
    assert [r['sentence_id'] for r in rank_scores(rows)]==['high','c','a','b','z']

def test_presentation_is_source_order_not_rank_order():
    ss=units();r=pack(ss,scores(ss,[0,10]),Tok(),256)
    assert r['selection_rank_ids']==['b','a']
    assert r['selected_unit_ids']==['a','b'] and r['text']=='A.\n\nB.'
    assert len(r['evidence_ids'])==r['evidence_tokens']<=256

def test_oversize_skipped_lower_rank_fits_and_negative_scores_not_thresholded():
    ss=units();ss[0].update(source_text='X'*300,end_char=300,token_count=300)
    r=pack(ss,scores(ss,[10,-10]),Tok(),256)
    assert r['selected_unit_ids']==['b']
    assert r['decisions'][0]['reason']=='SKIPPED_DOES_NOT_FIT'
    assert r['decisions'][1]['selected'] and r['decisions'][1]['score']==-10

def test_no_gold_outcome_or_phelp_used():
    ss=units();a=pack(ss,scores(ss,[1,2]),Tok(),256)
    for s in ss:s.update(gold='secret',f1=100,LOO=10,p_help=0)
    b=pack(ss,scores(ss,[1,2]),Tok(),256)
    assert a['evidence_ids']==b['evidence_ids'] and a['selection_rank_ids']==b['selection_rank_ids']

def test_duplicate_missing_or_changed_sources_rejected():
    ss=units();sc=scores(ss,[1,2])
    with pytest.raises(ValueError):pack(ss,sc[:1],Tok(),256)
    with pytest.raises(ValueError):pack(ss,sc+[sc[0]],Tok(),256)
    sc[0]['sentence_text']='altered'
    with pytest.raises(ValueError):pack(ss,sc,Tok(),256)

@pytest.mark.parametrize('budget',[0,128,512])
def test_unregistered_budget_fails(budget):
    with pytest.raises(ValueError):pack(units(),scores(units(),[1,2]),Tok(),budget)

def test_empty_and_actual_source_text():
    r=pack([],[],Tok(),1024)
    assert r['text']=='' and r['evidence_tokens']==0
    for b in [256,1024]:
        r=pack(units(),scores(units(),[1,2]),Tok(),b)
        assert all(u['source_text'] in r['text'] for u in r['units'])

def test_atomic_resume_incompatible_identity_and_scope(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'HOME',tmp_path)
    st=p.Store(tmp_path/'results/rows',{'model':'pinned'})
    st.put(['q'],{'label':p.LABEL,'answer':'saved'});st.put(['q'],{'label':p.LABEL,'answer':'saved'})
    assert st.get(['q'])['answer']=='saved'
    with pytest.raises(ValueError):st.put(['q'],{'label':p.LABEL,'answer':'overwrite'})
    with pytest.raises(ValueError):p.Store(st.path,{'model':'different'}).get(['q'])
    with pytest.raises(ValueError):p.save(tmp_path.parent/'out_of_scope.json',{})

def test_smoke_exactly_three_queries_full_cohort_not_resampled():
    ctx={'cohort':{'selected':{d:[dict(query_id=f'{i:02d}',group=f'g{i}') for i in range(50)] for d in p.DATASETS}}}
    assert sum(map(len,p.members(ctx,'smoke').values()))==3
    assert p.members(ctx,'full')==ctx['cohort']['selected']
    assert all(ms[0]['query_id']=='00' for ms in p.members(ctx,'smoke').values())

def test_full_generation_and_preparation_unlock_before_inputs(monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('must not open full-run inputs')
    monkeypatch.setattr(runner,'context',forbidden)
    with pytest.raises(ValueError):runner.generate_full()
    with pytest.raises(ValueError):runner.prepare_full()

def test_five_method_table_exact_columns_and_tied_wins():
    rows=[dict(dataset=d,budget=b,method=m,quality=.5 if m in p.CONTROLS else .6)
        for d in p.DATASETS for b in p.BUDGETS for m in p.METHODS]
    table,wins=comparison_table(rows)
    assert list(table)==['Dataset','Budget','FIXED','QBGF','BEA-V1','BEA-V2','MiniLM Reranker','Winner']
    assert len(table)==6 and wins[p.METHOD]==6
    for r in rows:r['quality']=.5
    _,wins=comparison_table(rows)
    assert all(v==6 for v in wins.values())
    with pytest.raises(ValueError):comparison_table(rows[:-1])

def test_wrong_pool_rejected_before_scoring(monkeypatch):
    from src.post_w12.bea_qbgf import resources
    from src.post_w12.bea_v2 import data
    ctx={'cohort':{'selected':{d:[dict(query_id='q',group='g')] for d in p.DATASETS}},
        'parent':{},'splits':{},'old':{},'v1':{}}
    e=dict(dataset='HotpotQA',query_id='q',inference={},cluster_id='g')
    monkeypatch.setattr(resources,'load_val',lambda *a:[e])
    bad=dict(source_hash=p.digest({}),lineage={'retriever':'BM25','candidate_ids':[],
        'candidate_content_hashes':[],'encoder':{}},pool=[])
    ctx['parent']={'models':{'SFR':{}}}
    monkeypatch.setattr(data,'problem',lambda *a:bad)
    with pytest.raises(ValueError,match='SFR top-10'):runner.load_inputs(ctx,'smoke')

def test_no_new_sentence_splitter_or_allocator_objective():
    path=Path(runner.__file__).parent/'minilm_reranker.py'
    source=path.read_text();tree=ast.parse(source)
    calls={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
    assert not calls & {'allocate','choose','solve','aspects','marginal','train','fit'}
    assert 'from src.week12.algorithm import serialize' in source
    assert 'p_help' not in source.lower().split('def pack(')[1]

def test_notebook_execution_requires_explicit_user_flags():
    n=p.read(p.HOME/'BaselineComparison_MiniLM.ipynb')
    code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    assignments={node.targets[0].id:node.value.value for node in ast.walk(ast.parse(code))
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
        and isinstance(node.value,ast.Constant)}
    # The user can explicitly enable the notebook; preserve their chosen flags.
    assert all(type(assignments.get(flag)) is bool for flag in ['RUN_FULL','RUN_SMOKE','RUN_READER_SMOKE'])
    assert 'if RUN_FULL:' in code

def test_historical_unicode_hash_conventions_are_checked_separately():
    from src.week11.cache import digest as unit_digest
    text='José — source evidence'
    assert unit_digest(text)!=p.digest(text)
    source=Path(runner.__file__).read_text()
    assert "candidate_content_hashes']==[digest(u['source_text'])" in source
    assert "u['hash']!=unit_digest(u['source_text'])" in source

@pytest.mark.parametrize('count',[1,8,10])
def test_exact_frozen_short_pool_accepted_without_padding(count):
    e=dict(inference={'question':'same'},cluster_id='g');member={'group':'g'}
    parent={'models':{'SFR':{'id':'pinned','revision':'same'}}}
    pool=[dict(unit_id=str(i),source_text=f'original source {i}') for i in range(count)]
    value=dict(source_hash=p.digest(e['inference']),pool=pool,lineage=dict(retriever='SFR',
        candidate_ids=[u['unit_id'] for u in pool],candidate_content_hashes=[p.digest(u['source_text']) for u in pool],encoder=parent['models']['SFR']))
    before=deepcopy(value)
    runner.check_frozen_identity(value,e,member,parent,'QASPER','q')
    assert value==before and len(value['pool'])==count
    value['lineage']['candidate_ids']=value['lineage']['candidate_ids'][1:]
    with pytest.raises(ValueError,match='candidate_ids'):runner.check_frozen_identity(value,e,member,parent,'QASPER','q')

@pytest.mark.parametrize('count',[0,11])
def test_empty_or_over_width_pool_still_fails(count):
    e=dict(inference={},cluster_id='g');parent={'models':{'SFR':{}}}
    pool=[dict(unit_id=str(i),source_text=str(i)) for i in range(count)]
    value=dict(source_hash=p.digest({}),pool=pool,lineage=dict(retriever='SFR',
        candidate_ids=[u['unit_id'] for u in pool],candidate_content_hashes=[p.digest(u['source_text']) for u in pool],encoder={}))
    with pytest.raises(ValueError,match='available_frozen_width'):
        runner.check_frozen_identity(value,e,{'group':'g'},parent,'QASPER','q')

def test_reader_cache_resume_produces_rows_without_loading_reader(tmp_path,monkeypatch):
    import src.week12.reader as reader_module
    import src.week11.evaluation as evaluation_module
    import src.post_w12.final_case_study.gold as gold_module
    monkeypatch.setattr(p,'HOME',tmp_path);monkeypatch.setattr(p,'OUT',tmp_path/'results')
    monkeypatch.setattr(runner,'OUT',tmp_path/'results')
    monkeypatch.setattr(runner,'token_counter',lambda ctx:Tok())
    def forbidden(*a,**kw):raise AssertionError('Cached answer must not load Qwen')
    monkeypatch.setattr(reader_module,'Reader',forbidden)
    monkeypatch.setattr(reader_module,'prompt_ids',lambda *a,**kw:[1,2,3])
    monkeypatch.setattr(evaluation_module,'evaluate',lambda *a:dict(f1=.5,exact_match=0.))
    monkeypatch.setattr(gold_module,'retention',lambda *a:{'gold_evidence_recall':None})
    parent=dict(models={'Qwen':{'id':'same','revision':'pinned'}},labels=['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])
    ctx=dict(parent=parent,record=dict(identity={'synthetic':True},identity_hash=p.digest({'synthetic':True})))
    _,selections,rows,answers=p.stores(ctx)
    e=dict(question='synthetic?',evaluation=dict(gold_label=None,gold_answer=['A']),task_type='freeform_qa',paper_id='paper')
    item=dict(dataset='QASPER',query_id='synthetic',member={'group':'paper'},example=e,
        problem=dict(source_hash='source',pool=[{'unit_id':'p'}],lineage={'candidate_ids':['p'],'candidate_content_hashes':['h']}))
    answer=dict(label=p.LABEL,prediction='A',raw_prediction='A',reader='BASE_FP16',
        model_hash=p.digest([parent['models']['Qwen'],'BASE_FP16',None]),prompt_hash=p.digest([1,2,3]),prompt_tokens=3,generated_tokens=1)
    answers.put(['QASPER','synthetic',p.digest([1,2,3])],answer)
    for b in p.BUDGETS:
        sel=pack(units(),scores(units(),[1,2]),Tok(),b)
        selections.put(['QASPER','synthetic',p.METHOD,b],dict(selection=sel,source_fingerprint='fp',behavior={'selected_units':2},source_hash='source'))
    runner._generate(ctx,[item],'full')
    for b in p.BUDGETS:
        row=rows.get(['QASPER','synthetic',p.METHOD,b])
        assert row['generated_answer']=='A' and not row['physical_generation_performed']
        assert row['conceptual_lm_calls']==1 and row['label']==p.LABEL
    # Repeated execution reuses immutable completed rows, including cross-budget reuse.
    runner._generate(ctx,[item],'full')

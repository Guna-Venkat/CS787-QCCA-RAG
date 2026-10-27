import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import pandas as pd
import pytest
from src.week11.protocol import load,guard,authorization,review,gate,scientific_config,output
from src.week11.cache import Cache,Store,atomic,digest,selection_key,file_hash
from src.week11.datasets import normalize,inference,source_identity
from src.week11.fixtures import examples
from src.week11.hierarchy import build,validate,bounded_passages
from src.week11.propositions import RulePropositions
from src.week11.budgets import FixtureTokenizer,allocate,evidence
from src.week11.retrievers import retrieve,passage_scores,BM25Index,ranked
from src.week11.selectors import select_scores,rank_units
from src.week11.embeddings import SyntheticEncoder,Embeddings
from src.week11.compressors import LLMLingua2,sara_guard
from src.week11.readers import Reader,FixtureReader
from src.week11.prompts import user_prompt
from src.week11.evaluation import evaluate,qa_pair,help_hurt,classification_metrics,evidence_metrics
from src.week11.bootstrap import paired
from src.week11.oracle import surface,capture
from src.week11.reporting import pareto
from src.week11.plotting import render
from src.week11.factorial import difference_in_differences
from src.week11.planning import action,shared_action,plan
from src.week11.generation import row,FIELDS

@pytest.fixture
def cfg():return load()

@pytest.fixture
def ex():return examples()[0]

@pytest.fixture
def units(ex,cfg):return build(ex['inference']['documents'],FixtureTokenizer(),cfg['hierarchy'])


def test_branch_guard_rejects_main(monkeypatch):
    import src.week11.protocol as p
    monkeypatch.setattr(p.subprocess,'check_output',lambda *a,**k:'main\n')
    with pytest.raises(RuntimeError,match='gagan'):guard()


@pytest.mark.parametrize('split',['test','TEST','official_test'])
def test_qasper_test_normalization_blocked(split):
    with pytest.raises(ValueError,match='TEST'):normalize('QASPER',split,'q','q',[])


def test_qasper_test_evaluation_blocked():
    with pytest.raises(ValueError,match='TEST'):evaluate('answer',dict(dataset='QASPER',split='test'))


def test_qasper_test_path_blocked_before_read(cfg):
    cfg['datasets']={'QASPER':dict(split='dev',path='missing/QASPER_test.jsonl')}
    with pytest.raises(ValueError,match='TEST'):source_identity(cfg)


def test_common_dataset_schema_and_no_gold_in_inference(ex):
    assert set('dataset split query_id document_id paper_id question gold_answer gold_label gold_evidence_ids gold_evidence_text document_text metadata answer_type task_type'.split())<=ex.keys()
    assert 'gold' not in json.dumps(inference(ex)).lower()
    assert ex['task_type']=='multihop_qa'


def test_scifact_full_corpus_not_gold_filtered():
    values=[e for e in examples() if e['dataset']=='SciFact']
    assert len(values)==3 and all(len(e['inference']['documents'])==2 for e in values)
    assert {e['gold_label'] for e in values}=={'SUPPORT','REFUTE','NOT_ENOUGH_INFO'}


def test_fixture_is_two_datasets_two_docs_three_queries():
    values=examples();assert len(values)==6
    for name in ['HotpotQA','SciFact']:
        g=[e for e in values if e['dataset']==name]
        assert len(g)==3 and len({d['document_id'] for e in g for d in e['inference']['documents']})==2
    assert all(e['split']=='dev' for e in values)


def test_hierarchy_exact_offsets_parentage(units,ex):
    validate(units,ex['inference']['documents'])
    assert {u['unit_type'] for u in units}=={'DOCUMENT','PASSAGE','SENTENCE','PROPOSITION'}


def test_corrupt_source_fails(units,ex):
    units[0]['source_text']='invented'
    with pytest.raises(ValueError,match='Source'):validate(units,ex['inference']['documents'])


def test_academic_sentence_offsets(cfg):
    from src.week11.datasets import document
    text='Dr. Smith used Fig. 2. Results improved by 3.5%. Replicate!'
    u=build([document('QASPER','d',text)],FixtureTokenizer(),cfg['hierarchy'])
    s=[x['source_text'] for x in u if x['unit_type']=='SENTENCE']
    assert s==['Dr. Smith used Fig. 2.','Results improved by 3.5%.','Replicate!']


def test_proposition_source_linkage_and_token_count(units):
    props=[u for u in units if u['unit_type']=='PROPOSITION']
    assert any('replication' in u['proposition_text'] for u in props)
    assert all(u['source_sentence_id'] and u['exact_source_support'] and u['proposition_token_count']==len(FixtureTokenizer().encode(u['source_text'])) for u in props)


def test_rule_propositions_are_exact():
    text='A improves; B declines.'
    out=RulePropositions().extract(text)
    assert len(out)==2 and all(text[r['start_char']:r['end_char']]==r['proposition_text'] for r in out)


def test_chunk_token_ceiling():
    text='one two three four five six'
    spans=bounded_passages(text,FixtureTokenizer(),2)
    assert [text[a:b] for a,b in spans]==['one two','three four','five six']


def test_bm25_matches_project_canonical():
    from src.week9.retrieval import bm25
    texts=['a a blue','blue red','green']
    assert np.allclose(BM25Index(texts).score('blue blue'),bm25('blue blue',texts))


def test_bm25_index_serialization():
    index=BM25Index(['x red','red blue'])
    assert np.allclose(index.score('red'),BM25Index.restore(index.serialize()).score('red'))


def test_ranking_ties_source_order(units):
    p=[u for u in units if u['unit_type']=='PASSAGE'][::-1]
    a=ranked(p,np.zeros(len(p)))
    assert a[0]['document_order']<a[1]['document_order']


def test_dense_retrieval_deterministic(units):
    p=[u for u in units if u['unit_type']=='SENTENCE']
    a,_=retrieve('red treatment',p,'MiniLM',SyntheticEncoder())
    b,_=retrieve('red treatment',p,'SFR',SyntheticEncoder())
    assert [u['unit_id'] for u in a]==[u['unit_id'] for u in b]


@pytest.mark.parametrize('name',['MiniLM','SFR'])
def test_frozen_embedding_adapter_identity(tmp_path,cfg,name):
    cfg['cache_root']=str(tmp_path)
    enc=Embeddings(cfg,name)
    assert enc.model is None and enc.spec==cfg['models'][name]
    assert enc.key('text','query')!=enc.key('text','document')
    enc.close()


@pytest.mark.parametrize('kind,expected',[('P0',[0.,1.]),('P1',[0.,1.]),('P2',[1.,1.]),('P3',[1.,1.])])
def test_passage_representation_math(kind,expected):
    q=np.array([1.,0.]);p=np.array([[0.,1.],[1.,0.]])
    s=[np.array([[1.,0.],[-1.,0.]]),np.array([[1.,0.]])]
    assert np.allclose(passage_scores(q,p,s,kind),expected)


def test_hybrid_deterministic_normalization():
    assert np.allclose(select_scores([1,3],[.2,.4],[2,4],'hybrid'),[0,1])
    assert np.allclose(select_scores([1,1],[.2,.2],[2,4],'hybrid'),[0,0])


def test_relevance_density_formula():
    assert np.allclose(select_scores([0,0],[.8,.8],[2,4],'density'),[.4,.2])


def test_lexical_selector_does_not_load_dense(units):
    result,_=rank_units('red',[u for u in units if u['unit_type']=='SENTENCE'],'lexical',None)
    assert len(result)>0


def test_budget_strict_prefix_and_actual_counts(units):
    toks={'Qwen':FixtureTokenizer(),'Mistral':FixtureTokenizer()}
    p=[u for u in units if u['unit_type']=='PASSAGE']
    assert allocate(p,1,toks)['selected_unit_ids']==[]
    selected=allocate(p,256,toks)
    assert max(selected['token_counts'].values())<=256 and selected['unused_budget']>=0


def test_source_order_restored(units):
    p=[u for u in units if u['unit_type']=='SENTENCE'][::-1]
    chosen=allocate(p,1024,{'Qwen':FixtureTokenizer()})
    assert chosen['selection_rank_ids']!=chosen['selected_unit_ids']
    assert chosen['units']==sorted(chosen['units'],key=lambda u:(u['document_order'],u['unit_id']))


def test_separator_tokens_and_all_readers_counted(units):
    class Characters:
        def encode(self,text,**kwargs):return list(text)
    p=[u for u in units if u['unit_type']=='SENTENCE']
    selected=allocate(p,30,{'Qwen':FixtureTokenizer(),'Llama':Characters()})
    assert len(selected['text'])<=30


def test_no_parent_child_double_exposure(units):
    result=allocate([u for u in units if u['unit_type']!='DOCUMENT'],1024,{'Qwen':FixtureTokenizer()})
    assert all(u['unit_type']=='PASSAGE' for u in result['units'])


def test_token_compressor_pending_honest():
    comp=LLMLingua2(dict(qualified=False))
    with pytest.raises(RuntimeError,match='pending'):comp.compress('source '*100,10,'q',{'Qwen':FixtureTokenizer()})
    assert comp.compress('source','FULL','q',{})['text']=='source'


def test_token_compressor_interface_mock(monkeypatch,tmp_path):
    class Compressor:
        def __init__(self,**kwargs):assert kwargs['use_llmlingua2']
        def compress_prompt(self,*args,**kwargs):return dict(compressed_prompt='red treatment')
    monkeypatch.setitem(sys.modules,'llmlingua',SimpleNamespace(PromptCompressor=Compressor))
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(tmp_path))
    out=LLMLingua2(dict(id='fake',revision='frozen',qualified=True)).compress('red treatment noisy text',3,'q',{'Qwen':FixtureTokenizer()})
    assert out['source_support'] is None and out['text']=='red treatment'


@pytest.mark.parametrize('reader',['Mistral','Qwen','Llama'])
def test_public_reader_uses_exact_pinned_spec(cfg,reader,monkeypatch):
    import src.week10.root_cause.runtime as runtime
    captured=[]
    monkeypatch.setattr(runtime,'TextReader',lambda spec:captured.append(spec) or SimpleNamespace(close=lambda:None))
    r=Reader(cfg,reader);r.close()
    assert captured==[cfg['models'][reader]]


def test_verification_prompt_is_task_specific():
    prompt=user_prompt('claim','evidence','verification')
    assert all(x in prompt for x in ['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])


@pytest.mark.parametrize('reader,dataset',[('Qwen','QASPER'),('Llama','QASPER'),('Mistral','SciFact')])
def test_sara_restricted(reader,dataset):
    with pytest.raises(ValueError,match='MISTRAL_ONLY'):sara_guard(reader,dataset)
    sara_guard('Mistral','QASPER')


def test_common_result_schema(cfg,ex,units):
    a=action(cfg);s=allocate([u for u in units if u['unit_type']=='SENTENCE'],512,{'Qwen':FixtureTokenizer()});s['candidate_pool_ids']=[]
    result=FixtureReader(cfg,'Qwen').generate(ex['question'],s['text'],ex['task_type'])
    baseline=dict(task_score=0.,prediction='unknown')
    identity=dict(config_hash='c',code_hash='s',model_hash='m',synthetic=True)
    r=row(cfg,'C',ex,a,s,result,baseline,identity)
    assert set(FIELDS)<=r.keys() and r['classification_accuracy'] is None and r['total_measured_latency'] is None


def test_specialist_help_hurt_cannot_use_public_baseline(cfg,ex,units):
    a=action(cfg,reader='Mistral',compressor='C3');selection=allocate([],0,{'Mistral':FixtureTokenizer()})
    result=dict(prediction='x',prompt_tokens=1,generated_tokens=1,generation_latency=0.)
    with pytest.raises(ValueError,match='same reader checkpoint'):
        row(cfg,'E',ex,a,selection,result,dict(reader_variant='PUBLIC_TEXT_READER',task_score=0.,prediction='x'),dict(synthetic=True))


def test_specialist_plan_includes_own_no_evidence_control(cfg):
    special=[a for a in plan(cfg,'E','QASPER') if a['compressor']=='C3']
    assert {a['granularity'] for a in special}=={'PASSAGE','NONE'}
    assert all(a['reader']=='Mistral' for a in special)


def test_hotpot_f1_and_em():
    assert qa_pair('The red treatment.','red treatment')==(1.,1.)
    assert qa_pair('yes','yes indeed')==(0.,0.)
    assert qa_pair('red','red blue')[0]==pytest.approx(2/3)


def test_qasper_metric_adapter_without_real_data():
    ex=normalize('QASPER','dev','fabricated','q',[],['red'])
    assert evaluate('red',ex)['f1']==1.


def test_classification_scores_and_invalid_label():
    ex=[e for e in examples() if e['dataset']=='SciFact'][0]
    assert evaluate('SUPPORT',ex)['classification_accuracy']==1
    assert evaluate('Maybe support',ex)['task_score']==0
    m=classification_metrics(['SUPPORT','REFUTE','NOT_ENOUGH_INFO'],['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])
    assert m['macro_f1']==1.


def test_help_hurt_qa_threshold():
    assert help_hurt(.5,.4)['HELP']
    assert help_hurt(.2,.4)['HURT']
    assert not help_hurt(.405,.4)['HELP']


def test_help_hurt_classification():
    assert help_hurt(1,0,True)['HELP'] and help_hurt(0,1,True)['HURT']


def test_exact_evidence_coverage(units,ex):
    passage=[u for u in units if u['unit_type']=='PASSAGE']
    assert evidence_metrics(passage,ex)['all_gold_recall']==1.


def test_proposition_coverage_does_not_penalize_boundary_whitespace(cfg):
    from src.week11.datasets import hotpot
    ex=hotpot([dict(_id='fabricated',question='q',answer='a',context=[['Doc',['A supports; B replicates.']]],supporting_facts=[['Doc',0]])])[0]
    units=build(ex['inference']['documents'],FixtureTokenizer(),cfg['hierarchy'])
    assert evidence_metrics([u for u in units if u['unit_type']=='PROPOSITION'],ex)['all_gold_recall']==1.


def test_D_selection_is_exact_C_cache_reuse(tmp_path,cfg,ex,monkeypatch):
    from src.week11.runner import Laboratory
    toks={k:FixtureTokenizer() for k in cfg['readers']}
    lab=Laboratory(cfg,[ex],toks,tmp_path,dict(embedding_weight_hash='fixture'),True)
    first=lab.selection(ex,action(cfg,granularity='SENTENCE',reader='Qwen'))
    lab.close()
    monkeypatch.setattr(lab,'encoder',lambda *a:(_ for _ in ()).throw(AssertionError('D must not recompute')))
    second=lab.selection(ex,action(cfg,granularity='SENTENCE',reader='Llama'),reuse_only=True)
    assert first==second


def test_D_missing_selection_refuses_recomputation(tmp_path,cfg,ex):
    from src.week11.runner import Laboratory
    lab=Laboratory(cfg,[ex],{'Qwen':FixtureTokenizer()},tmp_path,{},True)
    with pytest.raises(RuntimeError,match='reuse'):lab.selection(ex,action(cfg),reuse_only=True)


def test_specialist_is_not_a_public_reader_comparison():
    from src.week11.reporting import comparisons
    rows=[dict(r,reader='Mistral',compressor='C3',evidence_granularity='PASSAGE',selector='semantic',retriever='SFR',
                retrieval_width=10,strategy='FIXED') for r in outcomes()]
    assert comparisons(pd.DataFrame(rows),dict(bootstrap_resamples=10,seed=42)).empty


def test_H4_component_order_and_parent_provenance(tmp_path,cfg,monkeypatch):
    from src.week11.runner import Laboratory
    from src.week11.datasets import document
    docs=[document('HotpotQA',str(i),f'Source {i} red '+('red '*i)+'.',passages=[dict(start_char=0,end_char=len(f'Source {i} red '+('red '*i)+'.'))]) for i in range(6)]
    ex=normalize('HotpotQA','dev','fixture-h4','red',docs,['red'])
    cfg['hierarchical']['sentence_passages']=2
    lab=Laboratory(cfg,[ex],{k:FixtureTokenizer() for k in cfg['readers']},tmp_path,{},True)
    fake=SimpleNamespace(compress=lambda text,*a,**k:dict(text=text,source_support=None,notes='fabricated token adapter'))
    monkeypatch.setattr(lab,'token_compressor',lambda:fake)
    chosen=lab.selection(ex,action(cfg,granularity='HIERARCHICAL',strategy='H4',budget='FULL'))
    import re
    positions=[int(x) for x in re.findall(r'Source (\d)',chosen['text'])]
    assert positions==sorted(positions) and len(chosen['token_tail_fragments'])>0
    assert chosen['source_support'] is None and all(not f['exact_contiguous_span'] for f in chosen['token_tail_fragments'])
    lab.close()


@pytest.mark.parametrize('field',['config_hash','code_hash','model_hash','dataset_hash','selection_hash'])
def test_all_resume_hash_components_rejected(tmp_path,field):
    identity=dict(config_hash='a',code_hash='a',model_hash='a',dataset_hash='a',selection_hash='a')
    Store(tmp_path,identity)
    with pytest.raises(ValueError,match='Provenance'):Store(tmp_path,dict(identity,**{field:'changed'}))


def outcomes(dataset='QASPER'):
    return [dict(dataset=dataset,query_id=str(i),cluster_id='paper' if dataset=='QASPER' else str(i),task_score=.5,
        action_id='a',task_type='freeform_qa',actual_evidence_tokens=10,prompt_tokens=20,evidence_granularity='PASSAGE',
        reader='Qwen',budget=256,compressor='C0') for i in range(3)]


@pytest.mark.parametrize('dataset',['QASPER','HotpotQA'])
def test_paired_cluster_or_query_bootstrap(dataset):
    left=outcomes(dataset);right=[dict(r,task_score=.2) for r in left]
    result=paired(left,right,n=100,seed=42)
    assert result['point']==pytest.approx(.3) and result['ci_low']==pytest.approx(.3)


def test_bootstrap_rejects_unpaired_or_mixed_metrics():
    left=outcomes();right=left[:2]
    with pytest.raises(ValueError,match='Unpaired'):paired(left,right)
    with pytest.raises(ValueError):paired(left+outcomes('SciFact'),left+outcomes('SciFact'))


def test_pareto_detection():
    x=pd.DataFrame([dict(dataset='x',prompt_tokens=10,task_score=.5),dict(dataset='x',prompt_tokens=20,task_score=.4),dict(dataset='x',prompt_tokens=30,task_score=.7)])
    assert pareto(x).pareto.tolist()==[True,False,True]


def test_oracle_epsilon_cost_rule_and_action_distribution():
    rows=outcomes()+[dict(r,action_id='b',task_score=.505,actual_evidence_tokens=20) for r in outcomes()]
    result=surface(rows,.01)
    assert result['exact_oracle']==pytest.approx(.505) and result['epsilon_oracle']==pytest.approx(.5)
    assert result['exact_distribution']=={'b':3} and result['epsilon_distribution']=={'a':3}
    assert len(result['concentration'])==4 and not result['deployable']


def test_oracle_refuses_incomplete_surface():
    with pytest.raises(ValueError,match='Incomplete'):surface(outcomes()+[dict(outcomes()[0],action_id='b')])


def test_classification_epsilon_is_exact():
    rows=[dict(r,task_type='verification') for r in outcomes()]
    result=surface(rows,.01)
    assert result['epsilon']==0


def test_ocr_zero_headroom_honest():
    assert capture(.6,.4,.8)==pytest.approx(.5)
    assert capture(.5,.5,.5) is None


def test_cache_checksum_and_no_overwrite(tmp_path):
    c=Cache(tmp_path,'units',dict(version=1));c.put('q',dict(answer='a'))
    with pytest.raises(ValueError,match='overwrite'):c.put('q',dict(answer='b'))
    p=c.store.filename('q');obj=json.loads(p.read_text());obj['row']['answer']='corrupt';p.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match='Corrupt'):c.get('q')


def test_resume_identity_mismatch(tmp_path):
    Store(tmp_path,dict(model_hash='a',selection_hash='x'))
    with pytest.raises(ValueError,match='Provenance'):Store(tmp_path,dict(model_hash='b',selection_hash='x'))


def test_shared_selection_key_ignores_reader_but_not_budget(cfg,ex):
    a=action(cfg,reader='Qwen');b=action(cfg,reader='Llama')
    assert selection_key(ex,shared_action(a),cfg,{})==selection_key(ex,shared_action(b),cfg,{})
    b['budget']=256
    assert selection_key(ex,shared_action(a),cfg,{})!=selection_key(ex,shared_action(b),cfg,{})


def test_gold_changes_do_not_change_ranking_or_selection(cfg,ex,units):
    from src.week11.runner import Laboratory
    old=ex['evaluation'];ex['evaluation']={'gold_answer':['nonsense'],'gold_spans':[]}
    a,_=rank_units(ex['inference']['question'],[u for u in units if u['unit_type']=='SENTENCE'],'hybrid',SyntheticEncoder())
    ex['evaluation']=old
    b,_=rank_units(ex['inference']['question'],[u for u in units if u['unit_type']=='SENTENCE'],'hybrid',SyntheticEncoder())
    assert a==b


def test_full_opt_in_required():
    with pytest.raises(ValueError):authorization(full=True,explicit=False)
    with pytest.raises(ValueError):authorization(full=True,explicit=True,smoke=True)
    with pytest.raises(ValueError):authorization(smoke=True,limit=4)


def test_missing_upstream_freeze_blocks_full(cfg,tmp_path):
    cfg['output_root']=str(tmp_path)
    with pytest.raises(RuntimeError,match='freeze'):gate(cfg,'A')


def test_freeze_refuses_smoke(cfg,tmp_path):
    cfg['output_root']=str(tmp_path);atomic(output(cfg,'0')/'status.json',dict(synthetic=True,executable_complete=True))
    with pytest.raises(ValueError,match='smoke'):review(cfg,'0',{},'reason')


def test_manual_review_freeze_and_future_finalists(cfg,tmp_path):
    cfg['output_root']=str(tmp_path);atomic(output(cfg,'0')/'status.json',dict(synthetic=False,executable_complete=True,pending=[]))
    saved=review(cfg,'0',{},'Synthetic test of the review infrastructure, not a real phase')
    assert saved['branch']=='gagan' and saved['rationale']
    changed=copy.deepcopy(cfg);changed['finalist_run_ids']=['manually_later']
    assert scientific_config(cfg,'0')==scientific_config(changed,'0')
    assert scientific_config(cfg,'F')!=scientific_config(changed,'F')
    gate(changed,'A')


def test_empty_finalists_are_not_automatically_selected(cfg):
    assert cfg['finalist_run_ids']==[] and plan(cfg,'F','QASPER')==[]


def test_unknown_action_settings_fail_instead_of_silent_ignoring(cfg):
    with pytest.raises(ValueError,match='Unknown action'):action(cfg,unknown_model='silent switch')


def test_plots_from_saved_csv(tmp_path):
    p=tmp_path/'source.csv';pd.DataFrame([dict(dataset='FABRICATED',budget=256,task_score=.5),dict(dataset='FABRICATED',budget=512,task_score=.6)]).to_csv(p,index=False)
    figures=render(pd.read_csv(p),('fixture','source','line','budget','task_score',None),tmp_path,True)
    assert len(figures)==1 and all(Path(figures[0][k]).is_file() for k in ['png','pdf','data'])


def test_difference_in_differences():
    cells=[[dict(r,task_score=s) for r in outcomes()] for s in [.8,.5,.4,.2]]
    assert difference_in_differences(cells,n=100)['point']==pytest.approx(.1)


def test_notebook_defaults_safe():
    import nbformat,ast
    from src.week11.protocol import ROOT
    paths=list((ROOT/'notebooks/week11').glob('*.ipynb'));assert len(paths)==11
    for p in paths:
        notebook=nbformat.read(p,as_version=4);cell=next(c for c in notebook.cells if c.cell_type=='code')
        values={}
        for statement in ast.parse(cell.source).body:
            if isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Constant):values[statement.targets[0].id]=statement.value.value
        assert values['RUN_FULL'] is False and values['OVERWRITE'] is False
        assert 'display' in '\n'.join(c.source for c in notebook.cells)


def test_default_notebook_prefers_saved_real_results(cfg,tmp_path):
    from src.week11.runner import notebook_smoke
    cfg['output_root']=str(tmp_path)
    assert notebook_smoke(cfg,'C',True,False)
    atomic(output(cfg,'C')/'status.json',dict(executable_complete=True,synthetic=False))
    assert not notebook_smoke(cfg,'C',True,False)
    assert not notebook_smoke(cfg,'C',False,False)
    # An invalid full+smoke request still reaches the explicit authorization guard.
    assert notebook_smoke(cfg,'C',True,True)


def test_frozen_model_ids_match_week10(cfg):
    from src.week11.protocol import ROOT
    old=json.loads((ROOT/'configs/week10/root_cause_v1.json').read_text())
    assert all(cfg['models'][k]['revision']==old['models'][k]['revision'] for k in cfg['models'])


def test_historical_bytes_unchanged():
    from src.week11.protocol import ROOT
    p=ROOT/'results/week11/multigranular/v1/setup/historical_sha256.json'
    expected=json.loads(p.read_text());changed=[]
    for name,sha in expected.items():
        path=ROOT/name
        if not path.exists() or file_hash(path)!=sha:changed.append(name)
    assert not changed,'Historical files changed: '+str(changed[:10])

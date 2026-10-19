"""Isolated scientific fixtures. Never write mock objects into results/week9."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from src.week9.provenance import assert_inference_clean, digest, aligned
from src.week9.caching import RowCache, read_jsonl, atomic_json
from src.week9.config import check_test, load_config, load_frozen
from src.week9.chunking import sentences, token_windows, locate_passages, fixed_chunks, original_chunks
from src.week9.semantic_chunking import semantic_chunks
from src.week9.retrieval import rank_candidates, bm25, unit_vectors
from src.week9.selection import select, validate_selection
from src.week9.sane_adapter import select_synopses, fingerprints, reference_namespace
from src.week9.extraction import extract
from src.week9.faithfulness import audit_extraction, supported_evidence
from src.week9.revision import revise, revision_messages
from src.week9.cost_accounting import QueryCost
from src.week9.evaluation import choose_global, paired_paper_bootstrap
from src.week9.ke_surface import surface_analysis
from src.week9.adaptive_ke import headroom_gate, train_oof, sequential_stop, FEATURES
from src.week9.manifests import freeze


class Tokenizer:
    def __call__(self,text,**kwargs):
        import re
        spans=[(m.start(),m.end()) for m in re.finditer(r'\S+',text)]
        return {'offset_mapping':spans,'input_ids':list(range(len(spans)))}
    def encode(self,text,**kwargs):
        return list(range(len(text.split())))


class Encoder:
    tokenizer=Tokenizer()
    def encode(self,texts):
        return np.array([[1+text.count('apple'),1+text.count('zebra')] for text in texts],dtype=float)


def document():
    text='An apple grows. Another apple falls. A zebra runs. The zebra stops.'
    return dict(paper_id='p',source_document_id='raw-p',text=text,source_sha256=digest(text),
                sections=[dict(section='Results',char_start=0,char_end=len(text))])


def chunks(n=6):
    doc=document();lineage=locate_passages(doc,[doc['text']])
    base=original_chunks(doc,lineage,Tokenizer())[0]
    return [dict(base,chunk_id=f'variable-{i*7+3}',text=f'apple fact {i}') for i in range(n)]


def inference():
    return dict(question_id='q',paper_id='p',question='apple?',initial_draft='zebra')


def test_sentence_spans_preserve_every_source_character():
    for text in ['', 'A.', 'A.  B!\nC? trailing', '\nA sentence.\n\nSecond sentence.  ']:
        assert ''.join(text[a:b] for a,b in sentences(text))==text


def test_token_windows_cover_source_and_preserve_overlap():
    text='  a b  c d e f   '
    windows=token_windows(text,Tokenizer(),3,1)
    assert windows[0][0]==0 and windows[-1][1]==len(text)
    assert all(a2<=b1 for (a1,b1),(a2,b2) in zip(windows,windows[1:]))
    assert all(len(text[a:b].split())<=3 for a,b in windows)
    with pytest.raises(ValueError):
        token_windows(text,Tokenizer(),3,3)


def test_all_representations_keep_exact_lineage_and_offsets():
    doc=document();lineage=locate_passages(doc,[doc['text']])
    for values in [original_chunks(doc,lineage,Tokenizer()),fixed_chunks(doc,lineage,Tokenizer(),5,1),
                   semantic_chunks(doc,lineage,Tokenizer(),Encoder(),buffer=1,threshold=.3,max_tokens=8,overlap=2)]:
        assert values
        covered=set()
        for c in values:
            assert c['text']==doc['text'][c['char_start']:c['char_end']]
            assert c['source_document_id']=='raw-p' and c['paper_id']=='p'
            assert c['original_passage_ids'] and c['sentence_end']>c['sentence_start']
            covered.update(range(c['char_start'],c['char_end']))
        assert covered==set(range(len(doc['text'])))


def test_semrag_adjacency_threshold_and_final_flush():
    class E:
        def encode(self,texts):
            assert len(texts)==4
            return [[1,0],[1,0],[0,1],[0,1]]
    doc=document();lineage=locate_passages(doc,[doc['text']])
    values=semantic_chunks(doc,lineage,Tokenizer(),E(),buffer=0,threshold=.5,max_tokens=99,overlap=0)
    assert len(values)==2
    assert 'Another apple' in values[0]['text'] and 'zebra stops' in values[-1]['text']
    assert values[0]['char_end']==values[1]['char_start']
    assert values[0]['chunker_config']['algorithm']=='SemRAG-buffered-project-v1'


def test_semrag_symmetric_buffer_used_for_embeddings_only():
    seen=[]
    class E:
        def encode(self,texts):
            seen.extend(texts)
            return [[1,0]]*len(texts)
    doc=document();spans=sentences(doc['text']);lineage=locate_passages(doc,[doc['text']])
    values=semantic_chunks(doc,lineage,Tokenizer(),E(),buffer=1,threshold=.1,max_tokens=99,overlap=0)
    assert seen[0]==doc['text'][0:spans[1][1]]
    assert seen[1]==doc['text'][0:spans[2][1]]
    assert values[0]['text']==doc['text']


def test_missing_and_ambiguous_original_lineage_fail():
    with pytest.raises(ValueError,match='Cannot align'):
        locate_passages(document(),['unrelated missing passage'])
    d=document();d['text']='Repeat. Repeat.'
    with pytest.raises(ValueError,match='Ambiguous'):
        locate_passages(d,['Repeat.'])


@pytest.mark.parametrize('method',['BM25','SEMANTIC','RRF'])
def test_retrieval_is_paper_local_and_width_bounded(method):
    c=chunks()
    rows=rank_candidates('apple','p',c,Encoder(),method,40)
    assert len(rows)==6 and {r['chunk_id'] for r in rows}=={x['chunk_id'] for x in c}
    assert all(r['bm25_rank'] for r in rows)
    if method!='BM25':
        assert all(r['dense_rank'] for r in rows)
    with pytest.raises(ValueError,match='Cross-paper'):
        rank_candidates('apple','wrong',c,Encoder(),method,10)


def test_rrf_preserves_component_rank_formula():
    rows=rank_candidates('apple','p',chunks(),Encoder(),'RRF',10)
    for r in rows:
        assert r['retrieval_score']==pytest.approx(1/(60+r['bm25_rank'])+1/(60+r['dense_rank']))


@pytest.mark.parametrize('field',['gold_answer','gold_evidence_overlap','is_gold_evidence','true_delta_f1',
 'promoted_f1','oracle_k','oracle_candidate','target_action','gold_answers','base_f1'])
def test_inference_schema_rejects_leakage_recursively(field):
    with pytest.raises(ValueError,match='Forbidden'):
        assert_inference_clean({'nested':[{field:1}]})
    with pytest.raises(ValueError,match='Forbidden'):
        select(dict(inference(),**{field:'secret'}),chunks(),1,'DraftSemantic',Encoder())


def test_retrieval_rejects_labels_on_candidates():
    c=chunks();c[0]['gold_answer']='secret'
    with pytest.raises(ValueError,match='Forbidden'):
        rank_candidates('apple','p',c,Encoder(),'BM25',10)


@pytest.mark.parametrize('method',['BM25','QuerySemantic','DraftSemantic','QueryDraft'])
def test_selection_preserves_variable_identities_and_clamps_available(method):
    c=chunks(9)
    for n in [0,1,2,4,40]:
        selected=select(inference(),c,n,method,Encoder())
        validate_selection(c,selected,n)
        assert len(selected)==min(n,len(c))
        assert {r['chunk_id'] for r in selected}<={r['chunk_id'] for r in c}
    with pytest.raises(ValueError):
        validate_selection(c,[dict(c[0],chunk_id='invented')],1)


def test_sane_reference_helpers_preserve_identity_pad_and_prompt():
    c=chunks(9)
    class Client:
        def chat(self,messages,**kwargs):
            assert 'JSON list' in messages[0]['content']
            return {'content':'[7, 2, 2, 999]'}
    chosen,synopsis=select_synopses('apple',c,4,Client())
    assert [r['chunk_id'] for r in chosen]==[c[6]['chunk_id'],c[1]['chunk_id'],c[0]['chunk_id'],c[2]['chunk_id']]
    assert 'keywords=' in synopsis
    assert all(len(v)==64 for v in fingerprints().values())
    assert {'ATOMIC','COMPARATIVE','TEMPORAL'}<=set(reference_namespace()['BLUEPRINT_GUIDED_PROMPT'].split())


def test_sane_zero_budget_has_no_lm_call():
    class Client:
        def chat(self,*a,**k):
            raise AssertionError('STOP must not call LM')
    assert select_synopses('apple',chunks(),0,Client())[0]==[]


def test_support_spans_and_source_identity_are_checked():
    c=chunks(2)
    text='| support_span | structured_fact | source_chunk_id |\n|---|---|---|\n'
    text+=f'| apple fact 0 | invented claim | {c[0]["chunk_id"]} |\n'
    text+=f'| apple fact 1 | apple fact 1 | {c[0]["chunk_id"]} |\n'
    text+='| apple fact 0 | apple fact 0 | nonexistent |\n'
    audit=audit_extraction(text,c)
    assert [r['exact_span_supported'] for r in audit['rows']]==[True,False,False]
    assert audit['exact_span_support_rate']==pytest.approx(1/3)
    assert audit['unsupported_structured_fact_rate'] is None
    assert audit['rows'][0]['fact_entailment']=='unmeasured'
    assert 'invented claim' not in supported_evidence(audit)
    assert audit_extraction('invalid format',c)['invalid_format']


def test_extraction_q_vs_qd_is_a_controlled_prompt_difference():
    c=chunks(1);messages=[]
    class Client:
        def chat(self,messages: list,**kwargs):
            globals_ignored=None
            captured.append(messages)
            return {'content':f'| evidence_quote | source |\n|---|---|\n| apple fact 0 | {c[0]["chunk_id"]} |'}
    captured=[]
    query=extract(inference(),c,Client(),draft_aware=False)
    draft=extract(inference(),c,Client(),draft_aware=True)
    assert captured[0][0]==captured[1][0]
    assert 'Initial draft' not in captured[0][1]['content']
    assert 'Initial draft' in captured[1][1]['content']
    assert query['selected_chunk_ids']==draft['selected_chunk_ids']
    assert draft['audit']['exact_span_support_rate']==1


def test_stop_returns_base_draft_without_generation():
    class Client:
        def chat(self,*a,**k):
            raise AssertionError('STOP called generator')
    assert revise(inference(),'',Client())=='zebra'
    assert 'Initial answer' in revision_messages(inference(),'source',True)[0]['content']
    assert 'Initial answer' not in revision_messages(inference(),'source',False)[0]['content']


def test_cumulative_costs_equal_component_sums():
    cost=QueryCost(initial_draft_prompt_tokens=100,initial_draft_generated_tokens=10)
    for stage in ('selector','extraction','revision'):
        cost.add_call(stage,dict(prompt_tokens=20,generated_tokens=3))
    r=cost.to_dict()
    assert r['total_query_prompt_tokens']==160
    assert r['total_query_generated_tokens']==19 and r['num_lm_calls']==4
    assert r['final_prompt_tokens']==20 and r['measured_end_to_end_latency'] is None


def test_resume_never_duplicates_and_variable_candidates_roundtrip(tmp_path):
    path=tmp_path/'rows.jsonl';identity={'source_hash':'real-test-fixture'}
    c=chunks(9)
    with RowCache(path,identity) as cache:
        key=cache.key('q','p',[x['chunk_id'] for x in c],'test')
        cache.put(key,dict(question_id='q',paper_id='p',candidates=c))
        cache.put(key,{'should_not_replace':'existing row'})
    with RowCache(path,identity,resume=True) as cache:
        assert len(cache.rows)==1
        assert cache.rows[key]['candidates']==c
    assert len(read_jsonl(path))==1
    with pytest.raises(ValueError,match='Incompatible'):
        RowCache(path,{'changed':True},resume=True)


def test_resume_repairs_only_unterminated_final_write(tmp_path):
    path=tmp_path/'rows.jsonl';identity={'a':1}
    with RowCache(path,identity) as cache:
        cache.put('first',{'question_id':'q','paper_id':'p'})
    with path.open('ab') as f:
        f.write(b'{"incomplete":')
    with RowCache(path,identity,resume=True) as cache:
        cache.put('second',{'question_id':'q2','paper_id':'p'})
    assert len(read_jsonl(path))==2
    with path.open('a') as f:
        f.write('corrupted complete line\n')
    with pytest.raises(json.JSONDecodeError):
        RowCache(path,identity,resume=True)


def test_cache_lock_and_no_accidental_overwrite(tmp_path):
    path=tmp_path/'rows.jsonl'
    with RowCache(path,{'a':1}) as cache:
        cache.put('x',{})
        with pytest.raises(RuntimeError,match='Concurrent'):
            RowCache(path,{'a':1},True)
    with pytest.raises(FileExistsError):
        RowCache(path,{'a':1})


def test_missing_real_artifacts_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path/'missing.jsonl')
    with pytest.raises(FileNotFoundError):
        load_frozen(tmp_path/'missing.json')


@pytest.mark.parametrize('allow,path',[(False,None),(True,None),(False,'fake.json')])
def test_test_split_requires_both_explicit_guards(allow,path):
    with pytest.raises(PermissionError):
        check_test('test',allow,path)


def test_unfrozen_config_cannot_unlock_test(tmp_path):
    path=tmp_path/'unfrozen.json';atomic_json(path,{'frozen':False})
    with pytest.raises(ValueError,match='complete FIT-frozen'):
        check_test('test',True,path)


def test_freeze_rejects_smoke_partial_and_heldout(tmp_path):
    for split,scope,complete in [('fit','partial',True),('internal_val','full',True),('fit','full',False)]:
        with pytest.raises(ValueError):
            freeze(tmp_path/'frozen.json','a',{},dict(split=split,scope=scope),complete)


def test_global_tolerance_is_applied_once():
    rows=[dict(policy='a',f1=.42,total_query_prompt_tokens=200,complexity=1),
          dict(policy='b',f1=.418,total_query_prompt_tokens=150,complexity=2),
          dict(policy='c',f1=.416,total_query_prompt_tokens=50,complexity=0)]
    assert choose_global(rows)['policy']=='b'


def test_paired_bootstrap_rejects_misalignment_and_clusters_papers():
    rows=[dict(question_id=str(i),paper_id=str(i//2),f1=i/10) for i in range(6)]
    assert paired_paper_bootstrap(rows,rows,n=100)['ci95']==[0.,0.]
    with pytest.raises(ValueError,match='Unpaired'):
        paired_paper_bootstrap(rows,rows[:-1],n=10)
    with pytest.raises(ValueError,match='aligned'):
        aligned(rows,rows[:-1])


def surface_fixture(counts=(0,1,2,4)):
    return [dict(question_id=str(q),paper_id=str(q),evidence_k=k,f1=.5+(k==1)*.1,
        final_prompt_tokens=10+k,total_query_prompt_tokens=10+k,total_query_generated_tokens=2,help=k==1,hurt=False)
        for q in range(6) for k in counts]


def test_epsilon_oracle_is_cost_sensitive_and_analysis_only():
    matrix,summary=surface_analysis(surface_fixture())
    assert matrix.oracle_evidence_k.tolist()==[1]*6
    assert not summary['oracle_is_deployable']
    with pytest.raises(ValueError,match='Incomplete'):
        surface_analysis(surface_fixture()[:-1])


def test_adaptive_gate_skip_and_proceed():
    thresholds=dict(epsilon=.01,quality_gain=.005,token_savings=.15,quality_tolerance=.003)
    rows=surface_fixture((0,1,2,3,4))
    assert not headroom_gate(rows,thresholds)['proceed']
    for r in rows:
        r['f1']=.8 if r['evidence_k']==int(r['question_id'])%5 else .3
    assert headroom_gate(rows,thresholds)['proceed']


def test_verifier_uses_paper_grouped_oof_and_proposed_action_features():
    rows=[]
    for paper in range(10):
        for k in range(1,5):
            rows.append(dict(question_id=str(paper),paper_id=str(paper),evidence_k=k,
                candidate_chunk_id=f'c-{paper}-{k}',marginal_target=.1-k/20,marginal_cost=10,
                candidate_query_similarity=.1,candidate_draft_similarity=.2,candidate_tokens=10,
                accepted_count=k-1,proposed_prompt_tokens=10))
    actions=pd.DataFrame(rows)
    pred,fold_ids,models,final=train_oof(actions,5)
    assert np.isfinite(pred).all() and len(set(fold_ids))==5
    for _,g in actions.assign(fold=fold_ids).groupby('paper_id'):
        assert g.fold.nunique()==1
    decisions=sequential_stop(actions,np.tile([.1,.1,-.1,.2],10))
    assert all(r['evidence_k']==2 for r in decisions)
    assert decisions[0]['selected_chunk_ids']==['c-0-1','c-0-2']


def test_notebooks_have_safe_defaults_and_no_test_dropdown():
    root=Path(__file__).resolve().parents[2]
    for path in (root/'notebooks/week9').glob('*.ipynb'):
        nb=json.loads(path.read_text())
        text='\n'.join(''.join(c['source']) for c in nb['cells'])
        assert 'RUN_FULL = False' in text and 'SMOKE_LIMIT = 2' in text
        assert 'if RUN_FULL:' in text
        assert '--allow-test' not in text
        for header in ['PHASE','SCIENTIFIC QUESTION','PREREQUISITES','INPUT ARTIFACTS','EXPECTED OUTPUTS','CURRENT SPLIT','FULL RUN FLAG']:
            assert header in ''.join(nb['cells'][0]['source'])


def test_configs_freeze_budget_distinctions():
    root=Path(__file__).resolve().parents[2]
    for phase in 'abcdefgh':
        cfg=load_config(root/f'configs/week9/w9{phase}.yaml')
        assert cfg['initial_full_k']==4 and cfg['retrieval_widths']==[10,20,40]
        assert cfg['evidence_counts']==[0,1,2,4]


def test_local_xmistral_continuations_are_not_sliced_by_prompt_length():
    import torch
    from src.week9.backend import SaraBackend
    class Batch(dict):
        def __getattr__(self,name):
            return self[name]
        def to(self,device):
            return self
    class T:
        pad_token_id=0
        eos_token_id=2
        def __call__(self,text,**kwargs):
            return Batch(input_ids=torch.tensor([[5]*20]),attention_mask=torch.ones(1,20))
        def decode(self,ids,**kwargs):
            assert ids==[9,10,2]
            return 'actual answer'
    class M:
        config=SimpleNamespace(max_position_embeddings=128)
        def generate(self,**kwargs):
            return torch.tensor([[9,10,2]])
    backend=SaraBackend.__new__(SaraBackend)
    backend.device='cpu';backend.tokenizer=T();backend.model=M()
    out=backend.generate('long canonical prompt',8)
    assert out['content']=='actual answer' and out['generated_tokens']==3 and out['prompt_tokens']==20


def test_changed_cached_row_content_is_rejected(tmp_path):
    path=tmp_path/'rows.jsonl'
    with RowCache(path,{'source':1}) as cache:
        cache.put('key',{'question_id':'q','prediction':'original'})
    row=json.loads(path.read_text());row['prediction']='tampered'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='identity mismatch'):
        RowCache(path,{'source':1},resume=True)


def test_real_fit_canonical_overlap_regression_without_generation():
    from src.week9.data import load_records
    from src.week9.provenance import ROOT
    records,_=load_records(load_config(ROOT/'configs/week9/w9a.yaml'),'fit')
    assert records
    for row in records:
        inf=row['inference']
        assert inf['initial_draft']==row['canonical']['prediction']
        assert len(inf['selected_full_indices'])==4 and len(inf['compressed_indices'])==6
        assert row['canonical']['saved_generated_token_ids']
        assert_inference_clean(inf)


def test_canonical_prompt_encoder_and_presentation_order_regression():
    from src.week9.data import load_records
    from src.week9.backend import load_tokenizer
    from src.week9.provenance import ROOT
    from src.week7.w7e.prompt_builder import build_mixed_fidelity_sara_prompt
    cfg=load_config(ROOT/'configs/week9/w9a.yaml')
    records,_=load_records(cfg,'fit')
    tokenizer=load_tokenizer(cfg)
    for row in records[:2]:
        inf=row['inference']
        prompt,tokens,compressed,_=build_mixed_fidelity_sara_prompt(inf['question'],inf['passages'],
            inf['selected_full_indices'],[],inf['compressed_indices'],{},tokenizer)
        assert tokens==row['canonical']['context_tokens']
        assert compressed==sorted(inf['compressed_indices'])
        positions=[prompt.index(f'Document {i+1}.') for i in sorted(inf['selected_full_indices'])]
        assert positions==sorted(positions)
    assert cfg['encoder']=='sentence-transformers/all-MiniLM-L6-v2'


def test_production_has_no_fabricated_result_fallbacks():
    import ast
    from src.week9.provenance import ROOT
    for path in (ROOT/'src/week9').glob('*.py'):
        tree=ast.parse(path.read_text())
        # No constant positive metric tables, fake generators, or blanket catches
        # that could replace missing data with synthetic outcomes.
        assert 'synthetic' not in {n.name.lower() for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        for node in ast.walk(tree):
            if isinstance(node,ast.ExceptHandler):
                assert node.type is not None
                assert any(isinstance(n,ast.Raise) for n in ast.walk(node))


def test_ambiguous_substring_is_resolved_by_complete_original_partition():
    doc=document();doc['text']='Repeated fact. Other text. Repeated fact.'
    doc['sections']=[dict(section='s',char_start=0,char_end=len(doc['text']))]
    spans=locate_passages(doc,['Repeated fact. Other text.','Repeated fact.'])
    assert spans[0]['char_start']==0
    assert spans[1]['char_start']==27


def test_cached_client_keeps_gold_out_and_counts_cached_calls(tmp_path):
    from src.week9.backend import CachedClient
    class B:
        answer_max_tokens=8
        def render(self,messages):
            return '\n'.join(m['content'] for m in messages)
        def generate(self,prompt,max_tokens,retrieval_embeds=None):
            assert 'gold_answer' not in prompt
            return dict(content='answer',prompt_tokens=9,generated_tokens=2,latency=.1)
    with RowCache(tmp_path/'calls.jsonl',{'test':1}) as cache:
        for _ in range(2):
            cost=QueryCost(initial_draft_prompt_tokens=5)
            client=CachedClient(B(),cache,inference(),cost,'revision')
            assert client.chat([dict(role='user',content='evidence')])['content']=='answer'
            assert cost.to_dict()['total_query_prompt_tokens']==14
            assert cost.num_lm_calls==2
        assert len(cache.rows)==1


def test_phase_runner_bounds_test_and_wrong_config_are_rejected_without_loading_models():
    from src.week9.runner import parser, main
    with pytest.raises(SystemExit):
        parser('a').parse_args(['--split','test'])
    with pytest.raises(ValueError,match='Wrong phase'):
        main('b',['--config','configs/week9/w9a.yaml'])
    with pytest.raises(ValueError,match='bounds'):
        main('a',['--smoke-limit','0'])


def test_end_to_end_intervention_calls_and_frozen_evidence_are_controlled(tmp_path,monkeypatch):
    from src.week9 import runner
    monkeypatch.setattr(runner,'canonical_context',lambda *args:('canonical context [/INST]',None,8))
    # These are isolated unit fixtures, never written to the experiment directory.
    class B:
        answer_max_tokens=8
        tokenizer=Tokenizer()
        device='cpu'
        model=SimpleNamespace(dtype='unused')
        def render(self,messages):
            return '\n'.join(m['content'] for m in messages)
        def generate(self,prompt,max_tokens,retrieval_embeds=None):
            assert 'EVALUATION_SECRET' not in prompt
            if 'precise evidence extractor' in prompt:
                import re
                source=re.search(r'## Source ([^\n]+)',prompt).group(1)
                content=f'| evidence_quote | source |\n|---|---|\n| apple fact 0 | {source} |'
            else:
                content='apple answer'
            return dict(content=content,prompt_tokens=len(prompt.split()),generated_tokens=len(content.split()),latency=.1)
    rec=dict(inference=inference(),canonical={'saved_generated_token_ids':[1,2]},
        evaluation={'answers':['EVALUATION_SECRET'],'answers_reformatted':['EVALUATION_SECRET']})
    pool=dict(candidates=chunks(6),retrieval_width=10,retrieval_actual_count=6,retrieval_query_time=.01)
    cfg=dict(query_draft_alpha=.5,extraction_max_tokens=64)
    spec=dict(policy='D2',selector='DraftSemantic',evidence_k=1,representation='query',revision=False)
    with RowCache(tmp_path/'calls.jsonl',{'test':1}) as calls, RowCache(tmp_path/'ext.jsonl',{'test':1}) as ext:
        row=runner.run_intervention(rec,pool,spec,Encoder(),B(),calls,{},cfg,ext)
        assert row['prediction']=='apple answer' and row['num_lm_calls']==3
        assert row['initial_draft_generated_tokens']==2
        assert row['total_query_prompt_tokens']==8+row['extraction_prompt_tokens']+row['revision_prompt_tokens']
        frozen=row['extraction']
        rerun=runner.run_intervention(rec,pool,dict(spec,policy='F1',revision=True),Encoder(),B(),calls,{},cfg,ext,frozen)
        assert rerun['extraction']['raw_text']==row['extraction']['raw_text']
        assert rerun['extraction_prompt_tokens']==row['extraction_prompt_tokens']
        assert len(ext.rows)==1
        stopped=runner.run_intervention(rec,pool,dict(spec,evidence_k=0),Encoder(),B(),calls,{},cfg,ext)
        assert stopped['prediction']==rec['inference']['initial_draft'] and stopped['num_lm_calls']==1

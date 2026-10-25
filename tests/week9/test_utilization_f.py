"""W9-F frozen evidence, utilization behavior, freezes and safe execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from src.week9.provenance import digest,file_hash
from src.week9.caching import RowCache
from src.week9.cost_accounting import QueryCost
from src.week9.evaluation import annotate_outcome,paired_paper_bootstrap
from src.week9.senttopk_e3.sentences import inventory
from src.week9.utilization_f import prompts,protocol,analysis,runner


class Tokenizer:
    name_or_path='fixture'
    def encode(self,text,**kw):return text.split()


@pytest.fixture
def cfg():return protocol.config('configs/week9/w9f_utilization.yaml')


@pytest.fixture
def source():
    chunks=[dict(chunk_id='c',paper_id='p',text='Alpha is known. Beta is also known. Gamma matters.',char_start=0,char_end=51)]
    ss=inventory('q','p',chunks,Tokenizer())
    for i,s in enumerate(ss):s.update(presentation_rank=i+1,selected_rank=i+1,cosine=.5)
    block='\n\n'.join(s['text'] for s in ss)
    row=dict(question_id='q',paper_id='p',question='What is known?',initial_draft='Alpha',
        policy='E3-k3',sentence_k=3,coarse_evidence_k=2,actual_sentence_k=3,
        selected_chunk_ids=['c'],source_chunk_hashes={'c':digest(chunks[0]['text'])},selected_sentences=ss,
        selected_sentence_ids=[s['sentence_id'] for s in ss],evidence_text=block,evidence_hash=digest(block),
        prediction='Alpha and Beta',answer_output_hash=digest('Alpha and Beta'),row_content_hash='saved',
        evidence_tokens=len(block.split()),source_support_validation_rate=1.,
        initial_draft_generated_tokens=1,retrieval_query_time=.1,selection_latency=.2,
        tokenizer=dict(name_or_path='fixture',tokenizer_class='Tokenizer'))
    record=dict(inference=dict(question_id='q',paper_id='p',question=row['question'],initial_draft='Alpha'),
        evaluation=dict(answers=['Alpha and Beta'],answers_reformatted=['Alpha and Beta']))
    return row,record,chunks


def test_evidence_exact_identity_and_tampering(source):
    row,record,chunks=source
    proof=protocol.check_evidence(row,record,chunks)
    assert proof['sentence_text_hashes']==[s['text_hash'] for s in row['selected_sentences']]
    for field,value in [('question_id','other'),('paper_id','other'),('selected_sentence_ids',['bad']),
                        ('evidence_hash','bad'),('source_chunk_hashes',{'c':'bad'})]:
        bad=copy.deepcopy(row);bad[field]=value
        with pytest.raises(ValueError):protocol.check_evidence(bad,record,chunks)
    bad=copy.deepcopy(row);bad['selected_sentences'][0]['text']='invented'
    with pytest.raises(ValueError):protocol.check_evidence(bad,record,chunks)


def test_f0_alias_proves_rendered_prompt_model_and_call(source):
    row,_,_=source
    model=dict(model_id='fixture');token=dict(name_or_path='fixture');prompt='exact prompt'
    row.update(model=model,tokenizer=token,final_prompt_hash=digest(prompt),decoding=dict(do_sample=False,max_new_tokens=128))
    call=dict(question_id='q',paper_id='p',prompt_sha256=digest(prompt),content=row['prediction'],
        call_identity=dict(model=model,revision_mode='F0',evidence_ids=row['selected_sentence_ids'],decoding_hash=digest(dict(max_tokens=128,do_sample=False))))
    assert protocol.f0_proof(row,prompt,model,token,[call])['equivalent']
    for badprompt,badmodel,badcalls in [('different',model,[call]),(prompt,{'model_id':'other'},[call]),(prompt,model,[])]:
        with pytest.raises(ValueError):protocol.f0_proof(row,badprompt,badmodel,token,badcalls)


def test_prompts_distinct_frozen_capped_and_no_draft_in_f0():
    inf=dict(question='QUERY_TOKEN',initial_draft='DRAFT_TOKEN')
    outputs={p:prompts.messages(inf,'EVIDENCE_TOKEN',p)[0]['content'] for p in prompts.POLICIES}
    assert 'DRAFT_TOKEN' not in outputs['F0']
    for p in ('F1',)+prompts.F2_VARIANTS:
        assert all(v in outputs[p] for v in ('QUERY_TOKEN','DRAFT_TOKEN','EVIDENCE_TOKEN'))
        assert 'only the final concise answer' in outputs[p]
    assert 'Preserve the original answer unless' in outputs['F1']
    assert all('authoritative' in outputs[p] and 'remove unsupported' in outputs[p] for p in prompts.F2_VARIANTS)
    assert len(set(outputs.values()))==4 and len(prompts.F2_VARIANTS)==2
    assert prompts.definitions()==prompts.definitions()
    for p in prompts.POLICIES:
        assert prompts.definitions()[p]['prompt_hash']==digest(prompts.definitions()[p]['messages'])


def test_gold_payload_rejected_and_eval_never_used():
    with pytest.raises(ValueError):prompts.messages(dict(question='q',initial_draft='a',gold_answer='SECRET'),'e','F1')
    text=str(prompts.messages(dict(question='q',initial_draft='a'),'e','F2-A'))
    assert 'SECRET' not in text


def test_d0_is_skip_f0_is_saved_fresh_answer(source):
    row,_,_=source
    base=dict(question_id='q',paper_id='p',prediction='Alpha',num_lm_calls=1)
    d0=runner.alias(row,'D0',dict(d0={'q':base}))
    f0=runner.alias(row,'F0',dict(d0={'q':base}))
    assert d0['prediction']=='Alpha' and not d0['answer_generation_called'] and d0['evidence_tokens']==0
    assert f0['prediction']=='Alpha and Beta' and f0['evidence_hash']==row['evidence_hash']
    assert f0['f0_alias'] and f0['new_answer_calls']==0


def test_help_hurt_annotation_and_no_gold_in_output(source):
    _,record,_=source
    r=annotate_outcome(dict(prediction='Alpha and Beta',evidence_text='Beta'),record['inference'],record['evaluation'])
    assert r['help'] and not r['hurt'] and r['delta_f1']>0
    assert 'answers' not in r and 'answers_reformatted' not in r


def state_row(q,base,score,policy='F1',cost=10):
    return dict(question_id=q,paper_id='paper-'+q,policy=policy,initial_draft='original',prediction=policy,
        f1=score,initial_f1=base,delta_f1=score-base,help=score-base>.01,hurt=score-base<-.01,
        answer_changed=True,exact_match=0,final_prompt_tokens=cost,total_query_prompt_tokens=cost,
        total_query_generated_tokens=5,num_lm_calls=2,evidence_tokens=3,revision_latency=.1)


def test_rescue_full_recovery_lost_gain_and_denominators():
    f0={'a':state_row('a',.5,.2,'F0'),'b':state_row('b',.2,.6,'F0'),'c':state_row('c',.4,.4,'F0')}
    revision=[state_row('a',.5,.5),state_row('b',.2,.2),state_row('c',.4,.4)]
    b=analysis.behavior(revision,f0)
    assert b['partial_rescue_rate']==b['full_recovery_rate']==b['hurt_rescue_rate']==1
    assert b['lost_gain_rate']==b['diminished_gain_rate']==1 and b['help_retention_rate']==0
    assert analysis.behavior([revision[-1]],f0)['partial_rescue_rate'] is None


def test_transition_table_and_magnitudes():
    rows=[]
    for p in ('D0',)+prompts.POLICIES:
        rows += [state_row('a',.5,.2 if p=='F0' else .5,p),state_row('b',.2,.6 if p=='F0' else .2,p)]
    summaries,transitions,_=analysis.summarize(rows)
    s={r['policy']:r for r in summaries}
    assert s['F1']['hurt_reduction_vs_f0']==.5 and s['F1']['help_reduction_vs_f0']==.5
    assert s['F0']['mean_delta_among_help']==pytest.approx(.4)
    assert s['F0']['mean_delta_among_hurt']==pytest.approx(-.3)
    assert len(transitions)==15
    assert next(r for r in transitions if r['policy']=='F1' and r['f0_state']=='HURT')['full_recovery_rate']==1


def test_cost_first_tolerance_intervention_rule_no_d0(cfg):
    a=state_row('a',.2,.5,'F0',cost=100)
    b=state_row('a',.2,.498,'F1',cost=50)
    c=state_row('a',.2,.496,'F2-A',cost=10)
    assert analysis.choose([a,b,c],cfg)['policy']=='F1'
    with pytest.raises(ValueError):analysis.choose([dict(a,policy='D0')],cfg)


def test_material_hurt_tier_after_token_cost(cfg):
    a=dict(state_row('q',.2,.5,'F2-A'),hurt=.02)
    b=dict(state_row('q',.2,.5,'F2-B'),hurt=.08,total_query_generated_tokens=1)
    assert analysis.choose([a,b],cfg)['policy']=='F2-A'


def test_paper_bootstrap_pairing_and_reproducibility():
    a=[dict(question_id=str(i),paper_id=str(i//2),f1=.6) for i in range(6)]
    b=[dict(r,f1=.4) for r in a]
    bs=paired_paper_bootstrap(a,b,n=10000,seed=42)
    assert bs['cluster']=='paper_id' and bs['resamples']==10000
    assert bs['ci95']==pytest.approx([.2,.2])
    assert bs==paired_paper_bootstrap(a,b,n=10000,seed=42)
    with pytest.raises(ValueError):paired_paper_bootstrap(a,b[:-1])


def test_frozen_handoff_only_after_complete_and_no_controller(tmp_path,cfg):
    rows=[]
    for p in ('D0',)+prompts.POLICIES:
        for i in range(4):
            r=state_row(str(i),.4,.4 if p=='D0' else .45,p,cost=10 if p=='F0' else 20)
            r.update(evidence_hash='hash',selected_sentence_ids=['s'])
            rows.append(r)
    ident=dict(model={},config=cfg,input_hashes={},upstream_protocol_hash='e3',upstream_manifest_hash='manifest')
    smoke=tmp_path/'smoke';smoke.mkdir()
    analysis.analyze(rows,cfg,ident,smoke,complete=False)
    assert not (tmp_path/'frozen_w9f_intervention.json').exists()
    full=tmp_path/'fit';full.mkdir();(full/'results.jsonl').write_text('fixture\n')
    decision=analysis.analyze(rows,cfg,ident,full,complete=True)
    assert decision['intervention']=='F0'
    freeze=json.loads((tmp_path/'frozen_w9f_intervention.json').read_text())
    assert freeze['skip_action']=='D0' and freeze['evidence_policy']=='E3-k3'
    handoff=json.loads((tmp_path/'w9f_controller_handoff.json').read_text())
    assert handoff['fit_research_labels_only'] and not handoff['controller_trained_or_run']
    assert len(handoff['queries'])==4
    assert set(r['comparison'] for r in __import__('csv').DictReader((full/'bootstrap.csv').open())) >= {'F0 - D0','F1 - F0','F2-frozen - F1'}
    before=(tmp_path/'frozen_w9f_intervention.json').read_bytes()
    analysis.analyze(rows,cfg,ident,full,complete=True)
    assert before==(tmp_path/'frozen_w9f_intervention.json').read_bytes()


def test_answer_level_failure_is_explicit_not_extraction():
    assert runner.invalid_answer('')=='empty_answer_generation'
    assert runner.invalid_answer('<think>reason</think>answer')=='invalid_answer_wrapper'
    assert runner.invalid_answer('Reasoning: because\nAnswer: fact')=='invalid_answer_wrapper'
    assert runner.invalid_answer('A concise supported answer.') is None


@pytest.mark.parametrize('raw',['Alpha and Beta','','<think>leaked reasoning</think>'])
def test_generation_does_not_call_retrieval_or_gold_prompt(source,tmp_path,monkeypatch,raw):
    src,record,_=source
    class Backend:
        tokenizer=Tokenizer();device='cpu';model=SimpleNamespace(dtype='fixture');answer_max_tokens=128
        def identity(self):return {'model_id':'fixture'}
        def render(self,msgs):return str(msgs)
        def generate(self,prompt,max_tokens,retrieval_embeds):
            assert max_tokens==128 and 'GOLD_SECRET' not in prompt
            assert src['evidence_text'] in prompt and 'Initial answer: Alpha' in prompt
            return dict(content=raw,prompt_tokens=50,generated_tokens=3,latency=.1)
    monkeypatch.setattr(runner,'canonical_context',lambda *args:('context[/INST]',None,10))
    record['evaluation']=dict(answers=['GOLD_SECRET'],answers_reformatted=['GOLD_SECRET'])
    with RowCache(tmp_path/'calls.jsonl',{}) as calls:
        result=runner.generate(record,src,'F2-A',Backend(),calls,dict(embeddings={}))
    assert result['evidence_hash']==src['evidence_hash'] and result['num_lm_calls']==2
    assert not result['extraction_prompt_tokens'] and not result['extraction_generated_tokens']
    assert result['total_query_generated_tokens']==4
    if runner.invalid_answer(raw):
        assert result['prediction']==record['inference']['initial_draft'] and result['answer_fail_closed']
        assert result['failure_reason']==runner.invalid_answer(raw)


def test_identity_grid_rejects_extra_evidence(source):
    src,record,_=source
    rows=[]
    for p in ('D0',)+prompts.POLICIES:
        r=dict(question_id='q',paper_id='p',policy=p,initial_draft='Alpha',prediction='Alpha',num_lm_calls=1)
        if p!='D0':r.update(runner.source_fields(src))
        rows.append(r)
    bundle=dict(e3_rows={'q':src})
    analysis.validate_rows(rows,[record],bundle)
    rows[-1]['evidence_hash']='altered'
    with pytest.raises(ValueError,match='evidence'):analysis.validate_rows(rows,[record],bundle)


def test_preservation_and_overlap_are_descriptive():
    assert analysis.normalized('Alpha,  BETA!')==analysis.normalized('alpha beta')
    assert analysis.lexical('alpha beta','alpha gamma','beta')['added_tokens_in_evidence']==1
    assert analysis.lexical('alpha beta','alpha gamma','beta')['removed_answer_token_count']==1


def test_old_inputs_readonly_snapshot_guard(tmp_path):
    p=tmp_path/'e3.json';p.write_text('{}');before=(p.read_bytes(),p.stat().st_mtime_ns)
    bundle=dict(preserved={str(p):file_hash(p)})
    protocol.unchanged(bundle)
    assert before==(p.read_bytes(),p.stat().st_mtime_ns)
    p.write_text('changed')
    with pytest.raises(ValueError):protocol.unchanged(bundle)


def test_no_g_or_selection_invocation_in_new_runner():
    import ast
    source=Path(runner.__file__).read_text()
    tree=ast.parse(source)
    calls=[node.func.id for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name)]
    assert not set(calls)&{'train_oof','sequential_stop','prepare_query','rank','select','retrieve'}


def test_notebook_all_disabled_executes_without_generation(monkeypatch):
    import subprocess
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:calls.append((a,kw)))
    monkeypatch.setattr(subprocess,'check_output',lambda *a,**kw:'gagan\n')
    nb=json.loads(Path('notebooks/week9/week9f_revision.ipynb').read_text())
    ns={}
    for cell in nb['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'w9f_notebook','exec'),ns)
    assert not calls and not ns['RUN_F_FULL'] and not ns['RUN_F_SMOKE']


def test_smoke_selection_deterministic_and_small(cfg):
    records=[dict(inference=dict(question_id=str(i))) for i in range(10)]
    assert protocol.smoke_records(records,cfg)==protocol.smoke_records(list(reversed(records)),cfg)
    assert len(protocol.smoke_records(records,cfg))==2

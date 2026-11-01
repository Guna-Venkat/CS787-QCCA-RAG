"""Tiny fixtures for grounded E: no real GPU calls or historical artifact writes."""
import json
import re
from pathlib import Path
from types import SimpleNamespace
import pytest
from src.week9.grounded_e.evidence import locate_support,parse,extractive,tokens
from src.week9.grounded_e.protocol import check_chunks,e_sources
from src.week9.grounded_e.runner import cache_key,produce,verify_gate,validate_rows
from src.week9.grounded_e import prompts
from src.week9.caching import RowCache,atomic_json
from src.week9.provenance import digest,file_hash,source_hashes

class Tokenizer:
    name_or_path='tiny-tokenizer'
    def encode(self,text,**kwargs):return re.findall(r'\w+|[^\w\s]',text)

@pytest.fixture
def chunks():return [dict(chunk_id='s1',paper_id='p',text='Alpha is measured with BLEU. Beta is measured with ROUGE.')]


def test_support_exact_offsets_no_fuzzy(chunks):
    source=chunks[0]['text'];span='measured with BLEU'
    start,end,canonical=locate_support(source,span)
    assert source[start:end]==canonical==span
    assert locate_support(source,'measured  with BLEU') is None
    assert locate_support(source,'measured with BLEU!') is None
    assert locate_support(source,'') is None

@pytest.mark.parametrize('policy',['E1-Q','E1-QD','E2'])
def test_structured_parser_and_conservative_facts(chunks,policy):
    obj=dict(source_chunk_id='s1',support_span='Alpha is measured with BLEU.')
    obj.update(dict(normalized_claim='BLEU',evidence_type='answer_detail',relevance='direct') if policy=='E2' else dict(fact='BLEU'))
    result=parse(policy,json.dumps([obj]),chunks,'q',Tokenizer(),128)
    assert result['extraction_valid'] and result['items'][0]['source_paper_id']=='p'
    item=result['items'][0]
    assert chunks[0]['text'][item['char_start']:item['char_end']]==item['exact_source_text']
    assert result['accepted_exact_span_support_rate']==1
    assert result['evidence_tokens']<=128
    obj['normalized_claim' if policy=='E2' else 'fact']='unsupported addition'
    bad=parse(policy,json.dumps([obj]),chunks,'q',Tokenizer(),128)
    assert bad['fail_closed'] and not bad['text']

@pytest.mark.parametrize('raw',['nonsense','```json\n[]\n```','{}','[{"fact":"invented"}]','[{"source_chunk_id":"s1","support_span":"Alpha is measured with BLEU.","fact":"BLEU","extra":"x"}]'])
def test_malformed_fails_closed(chunks,raw):
    r=parse('E1-Q',raw,chunks,'q',Tokenizer(),128)
    assert r['fail_closed'] and r['invalid_format'] and not r['text']


def test_reject_invented_span_wrong_id_and_empty(chunks):
    for obj in [dict(source_chunk_id='wrong',support_span='Alpha',fact='Alpha'),dict(source_chunk_id='s1',support_span='invented',fact='invented')]:
        r=parse('E1-Q',json.dumps([obj]),chunks,'q',Tokenizer(),128)
        assert r['zero_exact_support'] and r['fail_closed'] and r['num_rejected_items']==1
    r=parse('E1-Q','[]',chunks,'q',Tokenizer(),128)
    assert r['empty_extraction'] and not r['extraction_valid']

@pytest.mark.parametrize('budget',[64,128])
def test_entire_evidence_budget_and_canonical_trimming(budget):
    cs=[dict(chunk_id='long-id',paper_id='p',text=' '.join(['Alpha']*300)+'.')]
    r=extractive(dict(question='Alpha',question_id='q'),cs,Tokenizer(),budget)
    assert r['extraction_valid'] and tokens(Tokenizer(),r['text'])<=budget
    for i in r['items']:assert cs[0]['text'][i['char_start']:i['char_end']]==i['support_span']


def test_schema_labels_and_oversized_item_rejected(chunks):
    obj=dict(source_chunk_id='s1',support_span=chunks[0]['text'],normalized_claim='BLEU',evidence_type='unbounded explanation',relevance='direct')
    r=parse('E2',json.dumps([obj]),chunks,'q',Tokenizer(),128)
    assert r['fail_closed']
    obj=dict(source_chunk_id='s1',support_span='Alpha is measured with BLEU.',fact='BLEU')
    class HugeTokenizer(Tokenizer):
        def encode(self,text,**kwargs):return list(text)*5
    assert parse('E1-Q',json.dumps([obj]),chunks,'q',HugeTokenizer(),64)['fail_closed']


def test_d_chunk_identity_not_substituted(chunks):
    row=dict(selected_chunk_ids=['s1'],source_chunks=chunks,evidence_k=2,specification=dict(selector='DraftSemantic',revision=False))
    assert check_chunks(row,row,row)==chunks
    bad={**row,'selected_chunk_ids':['other']}
    with pytest.raises(ValueError,match='identity'):check_chunks(row,bad,row)
    bad={**row,'source_chunks':[{**chunks[0],'text':'modified'}]}
    with pytest.raises(ValueError,match='identity'):check_chunks(row,row,bad)


def test_cache_policy_prompt_budget_and_sources(tmp_path,chunks):
    with RowCache(tmp_path/'e.jsonl',dict(sources=e_sources(),policy_names=prompts.POLICIES)) as cache:
        inf=dict(question_id='q',paper_id='p')
        base=cache_key(cache,inf,'E1-Q',chunks,'prompt-v1',128)
        assert base!=cache_key(cache,inf,'E1-QD',chunks,'prompt-v1',128)
        assert base!=cache_key(cache,inf,'E1-Q',chunks,'prompt-v2',128)
        assert base!=cache_key(cache,inf,'E1-Q',chunks,'prompt-v1',64)
        assert base!=cache_key(cache,inf,'E1-Q',[{**chunks[0],'text':'changed'}],'prompt-v1',128)


def test_fail_closed_skips_answer_generation_and_charges_extraction(tmp_path,chunks,monkeypatch):
    from src.week9.grounded_e import runner
    monkeypatch.setattr(runner,'canonical_context',lambda *a:('canonical',None,3))
    monkeypatch.setattr(runner,'annotate_outcome',lambda row,*a:row)
    class Backend:
        tokenizer=Tokenizer();device='cpu';model=SimpleNamespace(dtype='float16');answer_max_tokens=128
        n=0
        def identity(self):return dict(model_id='fixture',model_revision='fixture',dtype='float16',quantization='none')
        def render(self,messages):return json.dumps(messages)
        def generate(self,prompt,max_tokens,retrieval_embeds=None):
            self.n+=1
            return dict(content='[{"source_chunk_id":"s1","support_span":"invented","fact":"invented"}]',prompt_tokens=20,generated_tokens=10,latency=.1)
    backend=Backend()
    record=dict(inference=dict(question_id='q',paper_id='p',question='Which metric?',initial_draft='original'),canonical={},evaluation={},_d_cost=dict(retrieval_query_time=.1,selection_latency=.2))
    cfg=dict(max_evidence_tokens=128,extraction_max_tokens=256,max_new_tokens=128)
    seal=dict(model_hashes={},retrieval_config_hash='r',selector_config_hash='s')
    with RowCache(tmp_path/'calls.jsonl',{}) as calls,RowCache(tmp_path/'objects.jsonl',{}) as objects:
        r=produce(record,chunks,{},'E1-Q',backend,calls,objects,None,cfg,seal)
    assert backend.n==1 and r['prediction']=='original' and r['fail_closed']
    assert r['num_lm_calls']==2 and r['extraction_generated_tokens']==10 and r['revision_prompt_tokens']==0


def test_prompt_query_control_hides_draft(chunks):
    inf=dict(question='Q?',initial_draft='PRIVATE_DRAFT')
    assert 'PRIVATE_DRAFT' not in json.dumps(prompts.messages('E1-Q',inf,chunks,128))
    assert 'PRIVATE_DRAFT' in json.dumps(prompts.messages('E1-QD',inf,chunks,128))
    assert 'may be incomplete or wrong' in json.dumps(prompts.messages('E1-QD',inf,chunks,128))


def test_full_gate_rejects_missing_stale_or_tampered(tmp_path):
    with pytest.raises(ValueError,match='preflight'):verify_gate(tmp_path,{'v':1})
    p=tmp_path/'preflight/preflight_gate.json'
    artifact=tmp_path/'rows.jsonl';artifact.write_text('data')
    gate=dict(protocol_identity_hash=digest({'v':1}),implementation_checks_passed=True,artifact_hashes={str(artifact):file_hash(artifact)},warnings=['malformed'])
    atomic_json(p,gate)
    with pytest.raises(ValueError,match='acknowledge'):verify_gate(tmp_path,{'v':1})
    assert verify_gate(tmp_path,{'v':1},True)==gate
    with pytest.raises(ValueError,match='stale'):verify_gate(tmp_path,{'v':2},True)
    artifact.write_text('changed')
    with pytest.raises(ValueError,match='changed'):verify_gate(tmp_path,{'v':1},True)


def test_historical_global_bundle_remains_exact():
    seal=json.loads(Path('results/week9/patch_verification/w9b_cuda_memory_fix/a_source_compatibility.json').read_text())
    assert source_hashes()==seal['approved_sources']


def test_d_inputs_opened_read_only_and_sealed_separately(tmp_path,monkeypatch,chunks):
    from src.week9.grounded_e import protocol as p
    monkeypatch.setattr(p,'ROOT',tmp_path)
    upstream=tmp_path/'historical';directory=upstream/'extraction/fit';directory.mkdir(parents=True)
    frozen=upstream/'frozen';frozen.mkdir()
    primary=tmp_path/'results/week9/frozen/w9b_frozen_config.json';primary.parent.mkdir(parents=True);primary.write_text('{}')
    cfg=dict(output_root=str(upstream),grounded_output_root=str(tmp_path/'new-e'),max_evidence_tokens=128,extraction_max_tokens=256)
    for name in ('w9d_representation.jsonl','w9d_representation.identity.json'):(directory/name).write_text('historical data')
    for name in 'abc':(frozen/f'w9{name}_frozen_config.json').write_text('historical frozen '+name)
    oldcfg={'extraction_max_tokens':1024}
    (directory/'w9d_manifest.json').write_text(json.dumps(dict(complete=True,scope='full',split='fit',source_hashes={'unchanged':'sha'},config=oldcfg,config_hash=digest(oldcfg))))
    upstream_frozen={'a':dict(settings=dict(representation='FIXED-5',retriever='SEMANTIC',retrieval_width=20),settings_hash='a'),
        'b':dict(settings=dict(evidence_k=2)), 'c':dict(settings=dict(selector='DraftSemantic'),settings_hash='c',model_hashes={})}
    monkeypatch.setattr(p,'dependencies',lambda cfg,*a:upstream_frozen)
    monkeypatch.setattr(p,'load_frozen',lambda *a:dict(settings=dict(evidence_k=0)))
    monkeypatch.setattr(p,'source_hashes',lambda:{'unchanged':'sha'})
    record=dict(inference=dict(question_id='q',paper_id='p',initial_draft='draft'))
    monkeypatch.setattr(p,'load_records',lambda *a:([record],{}))
    rows=[]
    for policy in ('D0','D1','D2','D3'):
        rows.append(dict(question_id='q',paper_id='p',policy=policy,initial_draft='draft',prediction='draft',selected_chunk_ids=['s1'],source_chunks=chunks,evidence_k=2,specification=dict(selector='DraftSemantic',revision=False),row_content_hash=policy))
    monkeypatch.setattr(p,'load_phase',lambda *a:rows)
    oldfiles=[f for f in tmp_path.rglob('*') if f.is_file()]
    before={str(f):(f.read_bytes(),f.stat().st_mtime_ns) for f in oldfiles}
    bundle=p.inputs(cfg)
    assert bundle['chunks']['q']==chunks
    assert {str(f):(f.read_bytes(),f.stat().st_mtime_ns) for f in oldfiles}==before
    assert (tmp_path/'new-e/budget_128/protocol/d_input_freeze.json').is_file()


def test_protocol_identity_is_json_roundtrip_safe():
    from src.week9.grounded_e.protocol import identity
    obj=identity({'max_evidence_tokens':128,'extraction_max_tokens':256},{'seal':dict(input_hashes={},model_hashes={})})
    assert obj==json.loads(json.dumps(obj))


def test_e_notebook_disabled_path_executes_without_generation(monkeypatch):
    import subprocess
    calls=[]
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:calls.append(a))
    nb=json.loads(Path('notebooks/week9/week9e_extraction_faithfulness.ipynb').read_text())
    ns={}
    for cell in nb['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'grounded_e_notebook','exec'),ns)
    assert ns['RUN_FULL'] is False and ns['RUN_SMOKE'] is False and not calls


def test_full_analysis_emits_all_paired_intervals_without_point_estimate_claims(tmp_path,chunks):
    from src.week9.grounded_e.analysis import summarize,PAIRS
    from src.week9.grounded_e.evidence import extractive
    ev=extractive(dict(question='Alpha',question_id='q'),chunks,Tokenizer(),128)
    base=[];rows=[]
    for i in range(3):
        r=dict(question_id=str(i),paper_id='p'+str(i%2),f1=.5,exact_match=0,delta_f1=0,help=False,hurt=False,answer_changed=False)
        base.extend(dict(r,policy=p) for p in ('D0','D1','D2','D3'))
        rows.extend(dict(r,policy=p,evidence=ev,compression_ratio=.1) for p in prompts.POLICIES)
    result=summarize(rows,base,tmp_path,complete=True,seed=42)
    assert set(result['paired'])=={a+' - '+b for a,b in PAIRS}
    assert all(v['point_estimate']==0 and v['ci95']==[0,0] and v['cluster']=='paper_id' and v['resamples']==10000 for v in result['paired'].values())
    assert 'No requested superiority' in result['interpretation'][0]

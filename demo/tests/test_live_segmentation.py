"""Source-only segmentation checks. No models, GPU, scoring or research writes."""
from pathlib import Path
import pytest
from demo.live.segmentation import FROZEN,ADAPTED,sentence_spans,documents,annotate,inventory
from src.week9.senttopk_e3.sentences import spans as frozen_spans
from src.week11.hierarchy import build
from demo.tests.test_live import CharTokenizer,FakeResources
from demo.live.engine import run,input_key,parse_passages

@pytest.mark.parametrize('text',[
 'First fact. Second fact. Third fact.',
 'Dr. Smith described examples, e.g. group stages. It changed.',
 'The U.S. hosted games. The value was 2.96.',
 'A wrapped\nline remains prose. The next sentence follows.',
 'First paragraph\n\nSecond paragraph.',
])
def test_plain_text_parity(text):
    assert sentence_spans(text,FROZEN)==frozen_spans(text)
    assert sentence_spans(text,ADAPTED)==frozen_spans(text)

@pytest.mark.parametrize('citations',['[19]',' [19]','[19][20]','[19, 20]','[19–21]','[A]'])
def test_citation_boundaries_exact_source(citations):
    text=f'First fact.{citations} Second fact.[20] Third fact.[21]'
    parts=[text[a:b] for a,b in sentence_spans(text,ADAPTED)]
    assert parts==[f'First fact.{citations}','Second fact.[20]','Third fact.[21]']

def test_frozen_bug_reproduced_not_changed():
    text='First fact.[19] Second fact.[20] Third fact.[21]'
    assert len(frozen_spans(text))==len(sentence_spans(text,FROZEN))==1
    assert len(sentence_spans(text,ADAPTED))==3

def test_inline_citations_do_not_split_sentence():
    text='The format,[13][14] as approved later,[15] changed.'
    assert sentence_spans(text,ADAPTED)==[(0,len(text))]

def test_abbreviation_with_citation_stays_intact():
    text='Dr.[1] Smith lives in the U.S.[2] He works here.'
    # Frozen initials rules deliberately retain U.S. at a possible true boundary;
    # the adapter must not pretend to solve this linguistic ambiguity.
    assert sentence_spans(text,ADAPTED)==[(0,len(text))]

def test_markdown_heading_and_bullet_boundaries():
    text='# Format\nThe format changed.[19] Another fact.\n- Item one\n- Item two. Another item sentence.'
    parts=[text[a:b] for a,b in sentence_spans(text,ADAPTED)]
    assert parts==['# Format','The format changed.[19]','Another fact.','- Item one','- Item two.','Another item sentence.']

def test_exact_hierarchy_offsets_and_labels():
    passages=['# Heading\nA fact.[19] Another fact.[20]','- A list item\n- Another item']
    tok=CharTokenizer();units=annotate(build(documents(passages),tok,{'max_passage_tokens':256}),ADAPTED)
    rows=inventory(passages,units,tok,ADAPTED)
    assert len(rows)==5
    for u in units:
        if u['unit_type']=='SENTENCE':
            assert u['source_text']==passages[int(u['document_id'][1:])-1][u['start_char']:u['end_char']]
            assert u['splitter']==ADAPTED
            if u['boundary_kind']=='heading':assert u['display_kind']=='HEADING SEGMENT'
            if u['boundary_kind']=='list_item':assert u['display_kind']=='LIST SEGMENT'
    assert all(r['adapted_subsegment_count']==1 for r in rows)

def test_no_stale_results_or_rankings_between_modes():
    passages=['First fact.[19] Second fact.[20] Third fact.[21]'];cache={};r=FakeResources()
    a=run('fact',passages,'FIXED_SENTENCE',1024,'BM25',r,prepared_cache=cache,segmentation=FROZEN)
    b=run('fact',passages,'FIXED_SENTENCE',1024,'BM25',r,prepared_cache=cache,segmentation=ADAPTED)
    assert len(a['sentences'])==1 and len(b['sentences'])==3
    assert a['input_key']!=b['input_key'] and len(cache)==2
    assert a['sentences'][0]['display_kind']=='MULTI-SENTENCE SEGMENT'
    assert all(u['display_kind']=='SENTENCE' for u in b['sentences'])
    assert len(b['sentence_inventory'])==3

@pytest.mark.parametrize('method',['FIXED_SENTENCE','QBGF_FULL'])
def test_world_cup_inputs_source_only_pipeline(method):
    from demo.live.preset import load_preset
    text='\n---\n'.join(load_preset()['passages'])
    passages=parse_passages(text);r=run('How did the format change?',passages,method,256,'BM25',FakeResources(),True,generate_answer=False,scoring_mode='controlled_sfr' if method=='QBGF_FULL' else 'pipeline_native')
    assert len(passages)==15
    assert r['selection']['evidence_tokens']<=256
    assert r['answer'] is None
    assert len(r['sentence_inventory'])>len(passages)
    assert all(s['adapted_subsegment_count']==1 for s in r['sentence_inventory'])

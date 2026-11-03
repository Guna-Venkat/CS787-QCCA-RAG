"""Read-only end-to-end replay over a published complete illustrative pack."""
import pandas as pd
import streamlit as st
from .protocol import METHODS,RETRIEVERS,BUDGETS,DATASETS,LABELS,digest,OUT
from demo.components import note,unit_card,budget_meter
from demo.plots import attribution

# Presentation wording only; preserve frozen method IDs and protocol metadata.
LABELS={**LABELS, "PROPOSITION":"Fine-grained source spans"}

def available_methods(retriever,technical=False):
    return tuple(m for m in METHODS if technical or retriever!='BM25' or m!='QBGF_FULL')

def method_label(method):
    return LABELS[method]+(' · DEMO-ONLY REPRESENTATION' if method in ('PROPOSITION','MULTI_SENTENCE_2','MULTI_SENTENCE_4') else '')

def panel(query,condition,technical=False):
    r={**condition['result'],'result_type':'CURATED_ILLUSTRATION'};s=condition['selection'];units=condition['units']
    st.caption('CURATED EXAMPLE')
    if r['method'] in ('PROPOSITION','MULTI_SENTENCE_2','MULTI_SENTENCE_4'):st.caption('DEMO-ONLY REPRESENTATION')
    if r['method']=='QBGF_FULL' and r.get('retriever')=='BM25':st.caption('HISTORICAL NATIVE LEXICAL VARIANT');st.warning('Raw lexical relevance was used by this saved variant. Because its score scale differs substantially from cosine relevance, this condition is not a controlled BM25-vs-SFR QBGF comparison.')
    st.markdown('#### '+LABELS[r['method']]);budget_meter(r['evidence_tokens'],r['budget'])
    columns=st.columns(3);columns[0].metric('Correctness' if query['task']=='verification' else 'Answer F1',f"{r['task_score']:.3f}");columns[1].metric('Prompt tokens',r['prompt_tokens']);columns[2].metric('Generated tokens',r['generated_tokens'])
    st.markdown('**Saved answer**');st.write(r['prediction'])
    if r['raw_prediction']!=r['prediction']:
        with st.expander('Raw reader output'):st.write(r['raw_prediction'])
    st.caption(f"{len(units)} evidence units · Qwen3-8B")
    if r.get('exact_match') is not None:st.caption('Exact match '+str(r['exact_match']))
    st.caption('Per-claim correctness is not cohort macro-F1.' if query['task']=='verification' else 'Per-example answer token F1.')
    with st.expander('Exact reader evidence',expanded=False):
        if not units:st.info('NONE: no evidence is passed to the reader.')
        st.code(s['text'] or '(empty evidence)',language=None)
        for u in units:unit_card(u)
    if s.get('choices'):
        with st.expander('Actual QBGF allocation choices'):st.dataframe(pd.DataFrame(s['choices']),hide_index=True,width='stretch')
    if technical:
        with st.expander('Provenance / measured costs'):
            st.json({k:r[k] for k in ['prompt_hash','evidence_hash','source_hash','model_hash','generation_seconds','conceptual_lm_calls','exact_prompt_cache_reused']})
            st.caption('Generation time is not measured end-to-end latency. Exact cache reuse retains conceptual LM calls.')

def render(bundle,technical=False):
    m=bundle['manifest']
    st.subheader('Curated Research Examples')
    st.write('Explore real saved examples showing how retrieval, evidence granularity, budget allocation, and reader behavior differ across methods.')
    if technical:
        with st.expander('About this demo · methodology / provenance'):
            st.write('Outcome-curated VAL illustrations, not a representative benchmark or a new confirmation study. Original Research Findings are unchanged.')
            st.json(dict(questions=len(bundle['queries']),conditions=m['expected_rows'],scope=m['scope']))
    columns=st.columns(3)
    with columns[0]:ds=st.selectbox('Dataset',DATASETS,key='curated_dataset')
    with columns[1]:rt=st.selectbox('Retriever',RETRIEVERS,key='curated_retriever')
    with columns[2]:budget=st.selectbox('Evidence-token budget',BUDGETS,key='curated_budget')
    qs=m['queries'][ds]
    qid=st.selectbox('Curated question',qs,format_func=lambda q:bundle['queries'][ds+':'+q]['question'],key='curated_question_'+ds)
    query=bundle['queries'][ds+':'+qid];st.markdown('### '+query['question'])
    st.caption('Same reader, different evidence policies. Explore how the selected context changes the answer.')
    st.caption('Saved examples retain their original pipeline-native scoring. Use LiveDemo for controlled BM25-versus-SFR answer comparisons.')
    with st.expander('Gold reference · evaluation only'):st.write(query['gold'].get('gold_label') or query['gold'].get('gold_answer'))
    allowed=available_methods(rt,technical)
    for key in ('curated_left','curated_right'):
        if key in st.session_state and st.session_state[key] not in allowed:st.session_state[key]='RAW_SKIP' if key=='curated_right' else 'FIXED_SENTENCE'
    if rt=='BM25' and not technical:st.caption('Saved native lexical QBGF is available only in Researcher Mode; it is not a controlled retriever comparison.')
    c=st.columns(2)
    with c[0]:left=st.selectbox('Left representation',allowed,index=2,format_func=method_label,key='curated_left')
    with c[1]:right=st.selectbox('Right representation',allowed,index=allowed.index('QBGF_FULL') if 'QBGF_FULL' in allowed else allowed.index('RAW_SKIP'),format_func=method_label,key='curated_right')
    lr=query['conditions'][digest([rt,budget,left])];rr=query['conditions'][digest([rt,budget,right])]
    pools=query['pools'];pool=pools[rt]['pool'];hierarchy=pools[rt]['hierarchy']
    with st.expander('Retrieved candidate pools · compare retrievers'):
        cols=st.columns(3)
        for col,retriever in zip(cols,RETRIEVERS):
            with col:
                st.markdown('#### '+retriever)
                for p in pools[retriever]['pool']:
                    with st.expander(f"#{p['rank']} · {p['document_id']}"):
                        st.caption(f"Relevance {p['score']:.4f} · {p['token_count']} source tokens");st.write(p['source_text'])
        st.caption('Full candidate pools. BM25 scores are lexical; dense scores are cosine. No gold enters ranking. HotpotQA remains provided-context retrieval.')
    c=st.columns(2)
    with c[0]:panel(query,lr,technical)
    with c[1]:panel(query,rr,technical)
    st.subheader('Same source, every granularity')
    parent=st.selectbox('Source passage',pool,format_func=lambda p:f"Rank {p['rank']} · {p['document_id']}",key='curated_source_'+ds+'_'+qid+'_'+rt)
    children=[u for u in hierarchy if u['passage_id']==parent['unit_id']];selected={u['unit_id'] for r in (lr,rr) for u in r['units']};full_exposed=parent['unit_id'] in selected
    c=st.columns(3)
    with c[0]:st.markdown('#### Whole passage');unit_card(parent,selected=full_exposed)
    with c[1]:
        st.markdown('#### Exact sentences')
        for u in children:
            if u['unit_type']=='SENTENCE':unit_card(u,selected=full_exposed or u['unit_id'] in selected)
    with c[2]:
        st.markdown('#### Fine-grained source spans')
        st.caption('Deterministic punctuation-based subdivisions of exact source sentences.')
        for u in children:
            if u['unit_type']=='PROPOSITION':unit_card(u,selected=full_exposed or u['unit_id'] in selected)
    st.caption('SELECTED refers to either displayed context. Individual answer cards retain exact method-specific membership. Deterministic punctuation-based subdivisions of exact source sentences.')
    st.subheader('Real packing constraints')
    cost=parent['serialized_block_tokens'];st.write(f"This full passage costs **{cost}** serialized tokens against **{budget}** available.")
    st.warning('The whole passage cannot fit even in an empty evidence budget.') if cost>budget else st.info('The passage fits alone; other selected sources can still consume the remaining budget.')
    st.subheader('Representations and budgets')
    rows=[]
    for b in BUDGETS:
        for a in allowed:
            r=query['conditions'][digest([rt,b,a])]['result'];rows.append(dict({k:r[k] for k in ('budget','task_score','evidence_tokens','prompt_tokens','generated_tokens')},method=LABELS[r['method']],scope='DEMO-ONLY REPRESENTATION' if a in ('PROPOSITION','MULTI_SENTENCE_2','MULTI_SENTENCE_4') else 'HISTORICAL NATIVE LEXICAL VARIANT' if rt=='BM25' and a=='QBGF_FULL' else 'Curated example of research policy'))
    st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    st.subheader('What actually changed the answer?')
    method=st.radio('Inspect utility for',list(dict.fromkeys([left,right])),horizontal=True,format_func=method_label,key='curated_utility')
    units=query['conditions'][digest([rt,budget,method])]['units']
    if not units:st.info('No evidence units in this context; there is no unit-deletion test for NONE.')
    else:
        event=st.plotly_chart(attribution(units),width='stretch',key='curated_scatter',on_select='rerun',selection_mode=('points',))
        points=event.selection.points if event else [];ids=[u['unit_id'] for u in units]
        target=points[0]['customdata'][0] if points else None
        context=(ds,qid,rt,budget,method)
        if st.session_state.get('curated_focus_context')!=context or target in ids:
            st.session_state['curated_focus_context']=context;st.session_state['curated_focused']=target if target in ids else ids[0]
        focused=st.selectbox('Inspect source unit',ids,key='curated_focused')
        u=next(u for u in units if u['unit_id']==focused);unit_card(u,focus=True)
        st.write('Answer with evidence:',u['original_answer']);st.write('Answer after deletion:',u['ablated_answer'])
        st.caption('Deletion utility = original score − ablated score. A neutral effect does not prove irrelevance or that the reader ignored the source.')
    if technical:
        with st.expander('Collection integrity'):
            st.json(dict(manifest_hash=digest(m),source='demo/replay_data/v1/bundle.json',freeze_sha256=m['freeze_sha256'],scope=m['scope']))

"""Additional scientific axes, always derived from named primary artifacts."""
from html import escape
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from demo.constants import COLORS,DATASETS
from demo.data_loader import load
from demo.evidence_registry import (TASKS,RETRIEVERS,READERS,model_registry,retrieval_slice,reader_slice,
                                   matched_retriever_downstream,sara_case,trace_comparison,packing_facts)
from demo.components import source,note,unit_card
from demo.plots import finish


def task_card(dataset):
    title,description=TASKS[dataset]
    with st.expander(dataset+' · '+title):
        st.write(description)
        st.caption('Different tasks need different evidence relationships. Different task metrics are not averaged into a project score.')
        source('retrieval_manifest','freeze')


def retriever_explorer(query):
    st.subheader('Retriever Effect Explorer')
    st.caption('The selector can only work with the evidence that enters its candidate pool.')
    retriever=st.segmented_control('Retriever',RETRIEVERS,default='SFR',key='lab_retriever') or 'SFR'
    definitions={'BM25':'Sparse lexical retrieval: exact terminology and word overlap can be useful.',
                 'MiniLM':'Earlier dense semantic baseline; its full per-query traces are not retained here.',
                 'SFR':'Dense retrieval retained for the frozen main pipeline. It is not universally superior.'}
    st.info(definitions[retriever])
    with st.expander('Same query, different retrieved evidence',expanded=False):
        groups=trace_comparison(query);cols=st.columns(3)
        for col,rt in zip(cols,RETRIEVERS):
            with col:
                st.markdown('#### '+rt)
                rows=groups[rt]
                if not rows:
                    st.info('No retained query-level candidate trace. Aggregate retrieval results are available in Research Findings.')
                    continue
                complete={int(p['rank']) for p in rows}==set(range(1,11))
                st.caption(f'{len(rows)} known parents · '+('all 10 ranks recovered' if complete else 'partial retained trace'))
                for p in rows:
                    shared=', '.join(p['shared_with'])
                    label=('Also in recovered '+shared+' trace') if shared else 'Only in this retained trace; missing pools prevent a uniqueness claim'
                    overlap='Observed gold overlap' if p['gold_overlap'] is True else 'No overlap · saved full-passage test' if p['gold_overlap'] is False else 'Gold overlap unavailable'
                    with st.expander(f"#{int(p['rank'])} · {p['document_id']}"):
                        st.caption(f"{int(p['token_count'])} source tokens · score {p['score']:.4f}")
                        st.caption(label);st.caption(overlap+' · retrospective only')
                        st.write(p['source_text']);st.caption('Passage ID '+p['unit_id'])
        st.caption('These are recovered historical parent ranks/scores, not a new retrieval run. BM25 scores and semantic cosine scores have different scales. Gold overlap never enters retrieval.')
        source('units','profiles')
    with st.expander('Retriever identities'):
        st.json(model_registry());source('w10_protocol','retrieval_manifest')
    if retriever=='MiniLM':
        note('MiniLM query replay is unavailable. No candidate pool, selected context or downstream answer is fabricated; use the aggregate three-retriever comparisons in Findings.')
    return retriever


def packing_panel(query,pool,replays,budget):
    with st.expander('Why whole passages sometimes cannot fit',expanded=False):
        if not pool:st.info('No verifiable parent passage available for this query.');return
        oversize=[p for p in pool if p.get('serialized_block_tokens',0)>budget]
        chosen=st.selectbox('Inspect a real source passage',pool,index=pool.index(oversize[0]) if oversize else 0,
                            format_func=lambda p:f"Rank {int(p['rank'])} · source {p['document_id']}",key='packing_source')
        facts=packing_facts(chosen,budget)
        if not facts['available']:
            st.info('Exact serialized block cost unavailable. Run the CPU recovery command in README; no estimated cost is substituted.');return
        fig=go.Figure()
        fig.add_trace(go.Bar(y=['Full source passage','Evidence budget'],x=[facts['cost'],budget],orientation='h',
                            marker_color=['#f59e0b','#22c3ad'],text=[facts['cost'],budget],textposition='auto'))
        fig.update_xaxes(title='Exact serialized evidence tokens');fig.update_layout(showlegend=False)
        st.plotly_chart(finish(fig,220),width='stretch',key='packing_chart')
        st.success('Whole passage fits as a standalone unit at this budget.') if facts['fits'] else st.warning('Whole passage cannot fit even in an empty budget. RAW_SKIP would have to skip it.')
        st.caption('This is an exact standalone eligibility check, not a reconstructed historical allocation trace. A passage that fits alone may still be omitted after other sources consume the budget.')
        selected=[]
        for r in replays:
            if not r:continue
            units=[u for u in r['units'] if u.get('source') and u['source']['passage_id']==chosen['unit_id']]
            selected.append({'Saved method':r['profile']['method'],'Observed selected units from this passage':len(units),
                             'Standalone selected-unit token sum':sum(u['tokens'] for u in units),'Full context tokens':r['profile']['evidence_tokens']})
        st.dataframe(pd.DataFrame(selected),hide_index=True,width='stretch')
        st.caption('Smaller selected units can distribute evidence across sources; this example does not guarantee that finer granularity improves the answer.')
        source('units','profiles','code')


def sara_panel(key):
    rows=sara_case()
    fig=px.bar(pd.DataFrame(rows),x='retriever',y='quality',color='retriever',color_discrete_map=COLORS,
               text=[f"{r['quality']:.4f}" for r in rows],hover_data=['n_queries','exact_match','prompt_tokens'])
    fig.update_yaxes(title='QASPER answer F1');fig.update_layout(showlegend=False)
    st.plotly_chart(finish(fig,280),width='stretch',key=key)
    delta=next(r['quality'] for r in rows if r['retriever']=='SFR')-next(r['quality'] for r in rows if r['retriever']=='BM25')
    st.metric('SFR − BM25 · same saved SARA k5/n10 specialist',f'{delta:+.5f}')
    st.caption('Same paired QASPER VAL IDs, adapted Mistral/SARA checkpoint and k5/n10 interface. Descriptive specialist comparison; not reader- or budget-matched to Qwen 256/1024 and not proof of intrinsic latent superiority.')
    source('specialists')


def retrieval_findings(dataset,technical=False):
    st.markdown('### Retriever quality changes the evidence ceiling')
    note('If necessary evidence never enters the candidate pool, a selector cannot recover it from that pool. Recall still does not determine final answer quality by itself.')
    task_card(dataset)
    c=st.columns(2)
    with c[0]:gran=st.selectbox('Retrieval unit',['PASSAGE','SENTENCE','PROPOSITION'],key='coverage_granularity')
    with c[1]:k=st.select_slider('Retrieval depth',options=[1,2,4,6,8,10],value=10,key='coverage_k')
    rows=retrieval_slice(dataset,gran,k)
    m=st.selectbox('Coverage metric',['any_gold_recall','all_gold_recall','evidence_coverage'],key='coverage_metric')
    fig=px.bar(pd.DataFrame(rows),x='retriever',y=m,color='retriever',color_discrete_map=COLORS,
               text=[f'{r[m]:.3f}' for r in rows],hover_data=['n_queries','gold_metric_denominator','retrieved_tokens'])
    fig.update_yaxes(title=m,range=[0,1.1]);fig.update_layout(showlegend=False)
    st.plotly_chart(finish(fig),width='stretch',key='retrieval_coverage')
    st.caption('W11 pilot · same P0 representation and cutoff across retrievers · 50 sampled queries/dataset. Gold-metric denominators may differ from total queries. HotpotQA uses provided distractor contexts; no universal SFR claim.')
    if technical:st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
    source('retrieval_pilot','retrieval_manifest')
    with st.expander('Earlier W10 retrieval diagnosis'):
        st.dataframe(pd.DataFrame(load('w10')),hide_index=True,width='stretch')
        st.caption('Historical QASPER DEV, different cohort/mapping from W11/W12. This is not a matched cross-week recall improvement.')
        source('w10')
    st.markdown('#### Retriever → downstream effect · matched Week12 control')
    b=st.segmented_control('Matched downstream evidence budget',[256,1024],default=256,key='retrieval_downstream_budget') or 256
    rs=matched_retriever_downstream(dataset,b)
    diff=next(r['quality'] for r in rs if r['retriever']=='SFR')-next(r['quality'] for r in rs if r['retriever']=='BM25')
    st.metric('SFR − BM25 · same RAW_SKIP / Qwen / budget',f'{diff:+.5f}')
    fig=px.bar(pd.DataFrame(rs),x='retriever',y='quality',color='retriever',color_discrete_map=COLORS,text=[f"{r['quality']:.3f}" for r in rs],hover_data=['n_queries','evidence_tokens','prompt_tokens'])
    fig.update_yaxes(title='Macro-F1' if dataset=='SciFact' else 'Answer F1');fig.update_layout(showlegend=False)
    st.plotly_chart(finish(fig,290),width='stretch',key='retriever_downstream')
    st.caption('Same frozen VAL IDs, Qwen BASE_FP16, RAW_SKIP selector and evidence budget; retrieval route changes. No matched MiniLM downstream arm exists here. The W11 recall chart above is a different cohort, so its recall delta is not paired with these answer deltas.')
    if technical:st.dataframe(pd.DataFrame(rs),hide_index=True,width='stretch')
    source('extension')
    with st.expander('SARA retriever case study · specialist context'):
        sara_panel('sara_retrieval')
    with st.expander('Retrieval failure cases: what can be verified?'):
        st.info('Complete matched per-query BM25/MiniLM/SFR retrieval and gold-coverage shards were removed. No definite paired retrieval-miss case can be certified from the surviving partial traces. We do not turn absence from a partial trace into evidence of a retrieval failure.')
        st.write('Conceptual boundary: missing evidence → selector never sees it → changing granularity cannot restore that missing information. This is a principle, not a fabricated case study.')
        source('retrieval_pilot')


def reader_findings(dataset,technical=False):
    st.markdown('### Same evidence protocol, different reader')
    b=st.segmented_control('Reader panel budget',[256,1024],default=256,key='reader_budget') or 256
    rs=reader_slice(dataset,b)
    if dataset=='SciFact':
        st.warning('Mistral/Llama primary scores here are dominated by strict label-format failures. These scores cannot cleanly compare reader reasoning ability. Qwen-based final confirmation had zero invalid outputs.')
    fig=px.bar(pd.DataFrame(rs),x='evidence_granularity',y='quality',color='reader',barmode='group',
               color_discrete_map=COLORS,text=[f"{r['quality']:.3f}" for r in rs],hover_data=['n_queries','evidence_tokens','invalid_label_rate'])
    fig.update_yaxes(title='Macro-F1' if dataset=='SciFact' else 'Answer F1')
    st.plotly_chart(finish(fig),width='stretch',key='reader_effect')
    st.caption('W11 public-reader pilot · SFR, semantic selection, P0, C0 and shared source selection. The D runner requires reuse of reviewed C selections. Token counts differ by reader tokenizer; raw per-query hashes are no longer available for a fresh independent replay audit.')
    if dataset=='SciFact':
        st.dataframe(pd.DataFrame(rs)[['reader','evidence_granularity','quality','invalid_label_rate','n_queries']],hide_index=True,width='stretch')
    # The CSV parser preserves numeric budget as a float, so accept that representation too.
    pairs=[r for r in load('reader_pairs') if r['dataset']==dataset and r['budget'] in (b,str(b))]
    if technical:
        st.dataframe(pd.DataFrame(rs),hide_index=True,width='stretch')
        st.dataframe(pd.DataFrame(pairs),hide_index=True,width='stretch')
    source('reader_pilot','reader_manifest','reader_pairs','w11_runner','reader_compliance')
    interactions=load('reader_interactions');cross=sum(r['ci_low']<=0<=r['ci_high'] for r in interactions)
    note(f'{cross}/{len(interactions)} saved reader × granularity interaction intervals include zero. Reader scores differ, but a general interaction is not established. These intervals are exploratory, not final-confirmation tests.')
    source('reader_interactions')
    with st.expander('W10 public reader panel · frozen evidence-hash protocol'):
        st.dataframe(pd.DataFrame(load('reader_panel'))[['reader','action','n_queries','f1','exact_match','mean_prompt_tokens']],hide_index=True,width='stretch')
        st.caption('Same-text hash checks were required against frozen C evidence. Public G readers differ from adapted C-Mistral and SARA; do not attribute cross-system score gaps solely to architecture or latent vectors.')
        if technical:st.dataframe(pd.DataFrame(load('reader_panel_ci')),hide_index=True,width='stretch')
        source('reader_panel','reader_panel_ci','w10_protocol','w10_runner')


def complexity_and_protocol(dataset,data):
    st.markdown('### Does more adaptive mean better?')
    rows=[r for r in data['validation'] if r['dataset']==dataset and r['method'] in ('RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL')]
    from demo.plots import bars
    st.plotly_chart(bars(rows,['RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL'],'Macro-F1' if dataset=='SciFact' else 'Answer F1'),width='stretch',key='complexity_quality')
    st.markdown('**Whole passages → sentence policy → fixed hierarchy → adaptive per-passage fidelity**')
    note('Additional allocation flexibility did not produce monotonically better VAL quality. This is an algorithm-structure comparison, not a measured runtime or parameter-count complexity scale.')
    st.dataframe(pd.DataFrame(rows)[['method','budget','quality','evidence_tokens','prompt_tokens','n_queries']],hide_index=True,width='stretch')
    source('validation')
    st.markdown('#### Validation → frozen decision → reserved confirmation')
    freeze=load('freeze')
    st.markdown('Compare on VAL → predefined promotion criterion → retain simple fallback → freeze identities and cohorts → no more tuning → confirmation')
    st.success('Frozen policy: '+freeze['method']+' · no LoRA')
    st.write(freeze['rationale'])
    st.caption('The final policy is not a post-confirmation per-dataset winner. No fresh QASPER confirmation exists; the reserved cohort is now consumed.')
    source('freeze','contrasts','consistency')
    with st.expander('Why the project started with SARA'):
        st.write('The early question was whether adaptive text/latent representation budgets could improve SARA. Retrieval, representation and reader diagnostics led to a broader empirical study rather than treating compression as the only bottleneck.')
        sara_panel('sara_complexity')


def diagnostic_details(dataset):
    with st.expander('Gold evidence overlap versus observed utility'):
        rs=[r for r in load('gold_overlap') if r['dataset']==dataset]
        st.dataframe(pd.DataFrame(rs),hide_index=True,width='stretch')
        st.caption('Descriptive unit-level comparison. Unknown mapping cases are not silently counted as non-gold. Gold overlap is retrospective annotation matching, not a guarantee of correctness or a causal explanation. Unit lengths and within-query dependence limit simple comparisons.')
        source('gold_overlap')
    with st.expander('Redundancy & retrospective sufficiency'):
        rs=[r for r in load('retention') if r['dataset']==dataset]
        st.dataframe(pd.DataFrame(rs),hide_index=True,width='stretch')
        st.caption('Retrospectively chosen subsets are not deployable selectors. Relative retention is undefined when full evidence does not improve over NONE; only finite eligible queries are summarized. Read eligible_queries alongside available_queries. Ratios can exceed one and are not probabilities.')
        source('retention','profiles')


def saved_cases(dataset):
    with st.expander('Saved failure and utility case studies',expanded=False):
        cases=[r for r in load('qualitative') if r['dataset']==dataset]
        if not cases:
            st.info('No retained qualitative case export for this dataset. Numerical attribution remains available; no narrative failure case is fabricated.');return
        r=st.selectbox('Saved diagnostic case',cases,format_func=lambda r:f"{r['kind']} · {r['query_id']} · {r['budget']} tokens")
        st.write(r['question']);st.caption(f"VAL · {r['method']} · query {r['query_id']} · {r['interpretation']}")
        cols=st.columns(2);cols[0].write('Original answer: '+r['original_answer']);cols[1].write('After deleting the highlighted unit: '+str(r['unit']['ablated_answer']))
        unit=dict(r['unit'],text=next((x['text'] for x in r['selected_evidence'] if x['unit_id']==r['unit']['unit_id']),None))
        unit_card(unit,focus=True)
        st.caption(f"Original per-example score {r['unit']['full_score']:.3f}; after deletion {r['unit']['ablated_score']:.3f}. SciFact example scores are correctness, not cohort macro-F1. Automatic case categories are inspection aids, not human-certified reasoning explanations.")
        source('qualitative','units')


def dataset_overview():
    with st.expander('Why the datasets are not interchangeable'):
        cols=st.columns(3)
        for col,dataset in zip(cols,DATASETS):
            with col:
                st.markdown('**'+dataset+'**')
                title,description=TASKS[dataset];st.write(title);st.caption(description)
                st.caption('Three-class macro-F1' if dataset=='SciFact' else 'Answer F1 / exact match')
        source('retrieval_manifest','freeze')

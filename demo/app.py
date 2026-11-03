"""Presentation-first, read-only evidence allocation dashboard."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
import streamlit as st
from demo.constants import DATASETS,METHODS,BUDGETS,LABELS,EXPLANATIONS,COLORS
from demo.data_loader import load,freeze_check,verify_manifest,DataError
from demo.result_registry import findings,profiles,curated_ids,replay_index,get_query,method_replay,lookup
from demo.components import hero,note,source,budget_meter,unit_card,explanation
from demo.plots import bars,deltas,attribution
from demo.evidence_registry import recovered_pool,retriever_scope
from demo.research_extensions import (task_card,retriever_explorer,packing_panel,retrieval_findings,reader_findings,
                                      complexity_and_protocol,diagnostic_details,saved_cases,sara_panel,dataset_overview)

st.set_page_config(page_title='Evidence Lab · CS787',page_icon='◈',layout='wide',initial_sidebar_state='auto')
st.markdown('<style>'+ (ROOT/'demo/assets/styles.css').read_text()+'</style>',unsafe_allow_html=True)
try:
    freeze_check();verify_manifest();data=findings();index={}
except (DataError,FileNotFoundError,ValueError) as exc:
    st.error('Primary artifact verification failed. Results are not being substituted. '+str(exc));st.stop()

try:
    index=replay_index()
except (DataError,FileNotFoundError,ValueError):
    index={}  # A missing/stale historical cache cannot disable a new self-contained pack.

with st.sidebar:
    st.markdown('### ◈ EVIDENCE LAB')
    st.caption('A study of context, not a chatbot.')
    presentation=st.toggle('Presentation Mode',value=True)
    technical=st.toggle('Researcher Mode',value=False)
    st.divider()
    st.caption('FROZEN FINAL POLICY')
    st.markdown('**SFR → FIXED_SENTENCE → Qwen3-8B**')
    st.caption('Top-10 candidates · BASE_FP16 · no LoRA\n\n256 / 1024 evidence tokens')
    with st.expander('Model & method identities'):
        st.write('Retriever: Salesforce/SFR-Embedding-Mistral')
        st.write('Reader: Qwen/Qwen3-8B')
        st.write('QBGF is the experimental adaptive allocation method, not the universally promoted final system.')
        source('freeze','freeze_copy')
    st.divider()
    st.caption('Curated examples run on CPU. LiveDemo models are optional and load only on request.')
    st.caption('GRANULARITY SPECTRUM')
    st.caption('Passage → multiple sentences → sentence → fine-grained unit. Finer does not mean better.')
    st.caption('Research files stay read only. Live inputs remain in session memory.')

if st.session_state.get('workspace') not in (None,'Curated Demo','LiveDemo','Research Findings','Research Journey'):
    st.session_state['workspace']='Curated Demo'
hero(compact=st.session_state.get('workspace','Curated Demo')!='Curated Demo')
with st.expander('Three findings to take away',expanded=False):
    cols=st.columns(3)
    cols[0].markdown('**Query-conditioned allocation matters.**\n\nValidation ablation within a query-dependent retrieved pool.')
    cols[1].markdown('**Fusion is not universally superior.**\n\nMixed options did not establish a consistent advantage.')
    cols[2].markdown('**Relevance is not utility.**\n\nWeak, context-dependent alignment in post-hoc deletion analysis.')
    source('primary','fusion','correlations')
with st.expander('Five different questions along the evidence pathway'):
    st.markdown('| Stage | Question |\n| --- | --- |\n| Retrieval | Which source evidence enters the candidate pool? |\n| Ranking | Which candidates appear most relevant? |\n| Granularity | In what form and level of detail is evidence represented? |\n| Allocation | How is the finite token budget distributed? |\n| Reader utilization | Which exposed evidence changes the answer or task score? |')
    source('code','w10_runner','w11_runner')
mode=st.segmented_control('Workspace',['Curated Demo','LiveDemo','Research Findings','Research Journey'],default='Curated Demo',selection_mode='single',key='workspace')


def technical_table(rows,*sources):
    if technical:
        with st.expander('Exact values, uncertainty & cohort sizes'):
            st.dataframe(pd.DataFrame(rows),hide_index=True,width='stretch')
            st.caption('Empty / — fields mean unmeasured or unavailable, never zero.')
    source(*sources)


def research_findings():
    st.subheader('What survived the experiments?')
    st.caption('FROZEN RESEARCH RESULT')
    st.caption('Validation mechanisms and reserved confirmation are labelled separately. QA F1 and SciFact macro-F1 are never pooled.')
    detail=st.radio('Detail',['Simple View','Technical View'],horizontal=True)
    ds=st.segmented_control('Result dataset',DATASETS,default='SciFact',key='findings_dataset')
    task_card(ds)
    dataset_overview()
    metric='Three-class macro-F1' if ds=='SciFact' else 'Answer F1'
    tabs=st.tabs(['01 · Evidence helps','02 · Granularity × budget','03 · Query conditioning','04 · Fusion ablation','05 · Relevance ≠ utility','06 · Retriever effects','07 · Reader effects','08 · Decision & complexity'])
    with tabs[0]:
        st.markdown('### Evidence helps the frozen reader')
        if ds=='QASPER':
            rows=[r for r in data['validation'] if r['dataset']==ds];st.info('QASPER has validation only. Historical TEST is closed; no fresh confirmation exists.')
            st.plotly_chart(bars(rows,['NONE','FIXED_SENTENCE'],metric),width='stretch',key='evidence_bars')
            technical_table(rows,'validation')
        else:
            rows=[r for r in data['confirmation'] if r['dataset']==ds]
            cs=[r for r in data['contrasts'] if r['dataset']==ds and r['right']=='NONE']
            st.caption('Reserved confirmation · '+str(int(rows[0]['n_queries']))+' queries · '+str(int(rows[0]['n_groups']))+' groups')
            cols=st.columns(2)
            for col,r in zip(cols,cs):col.metric(f"FIXED_SENTENCE − NONE · {int(r['budget'])}",f"{r['point']:+.3f}",help=f"Nominal 95% CI [{r['ci_low']:.5f}, {r['ci_high']:.5f}]; Holm p={r['holm_p']:.5f}")
            st.plotly_chart(bars(rows,['NONE','FIXED_SENTENCE'],metric),width='stretch',key='confirmed_evidence')
            note('The frozen evidence-assisted pipeline improved over NONE in all four confirmation cells after the registered primary-family correction. This does not isolate retrieval from selection.')
            technical_table(cs,'confirmation','contrasts')
    with tabs[1]:
        st.markdown('### What did we learn about granularity?')
        rows=[r for r in data['confirmation' if ds!='QASPER' else 'validation'] if r['dataset']==ds]
        st.caption('Reserved confirmation' if ds!='QASPER' else 'Validation · no fresh QASPER confirmation')
        st.plotly_chart(bars(rows,['RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL'],metric),width='stretch',key='granularity_bars')
        if ds=='SciFact':
            r=next(r for r in data['contrasts'] if r['dataset']==ds and r['budget']==256 and r['right']=='RAW_SKIP')
            note(f"At 256, sentence minus passage = {r['point']:+.3f} macro-F1, nominal CI [{r['ci_low']:.3f}, {r['ci_high']:.3f}]. At 1024 passages have the higher point estimate; the paired difference remains uncertain.")
        else:note('Granularity is task- and budget-dependent. Broader context can recover competitiveness; finer evidence is not guaranteed to win for every example.')
        st.caption('RAW_SKIP is budget-matched whole-passage RAG. Some whole passages still cannot fit at a tight budget. H3 point estimates are descriptive controls.')
        technical_table(rows,'confirmation' if ds!='QASPER' else 'validation','contrasts')
    with tabs[2]:
        st.markdown('### Removing query-conditioned allocation lowered quality in all six VAL cells')
        rows=[r for r in data['validation'] if r['dataset']==ds]
        st.plotly_chart(bars(rows,['QBGF_FULL','QBGF_NO_QUERY'],metric),width='stretch',key='query_bars')
        cs=[r for r in data['primary'] if r['right']=='QBGF_NO_QUERY']
        st.plotly_chart(deltas(cs,'QBGF_FULL − QBGF_NO_QUERY · task-specific score'),width='stretch',key='query_ci')
        st.caption('Rows use their own dataset metric; QA F1 and macro-F1 effect sizes are not one pooled scale.')
        note('Upstream SFR retrieval remained query-dependent. This isolates allocation conditioning, not query-free retrieval. It is a validation mechanism result, not a confirmation test.')
        technical_table(cs,'validation','primary')
    with tabs[3]:
        st.markdown('### More representation choices did not consistently improve answers')
        rows=[r for r in data['validation'] if r['dataset']==ds]
        st.plotly_chart(bars(rows,['QBGF_FULL','QBGF_SENTENCE_ONLY'],metric),width='stretch',key='fusion_bars')
        cs=[r for r in data['fusion'] if r['right']=='QBGF_SENTENCE_ONLY']
        st.plotly_chart(deltas(cs,'QBGF_FULL − QBGF_SENTENCE_ONLY · task-specific score'),width='stretch',key='fusion_ci')
        note('Query conditioning helped its allocation mechanism; allowing both sentences and whole passages did not establish a consistent fusion advantage. QBGF does have a predefined HotpotQA @1024 confirmation win over FIXED_SENTENCE.')
        technical_table(cs,'fusion','validation','contrasts')
    with tabs[4]:
        st.markdown('### Relevance is an imperfect proxy for downstream utility')
        counts=data['unit_counts'];total=sum(counts.values());cols=st.columns(3)
        for col,category in zip(cols,['HELPFUL','NEUTRAL','HARMFUL']):col.metric(category+' · individual deletions',f'{100*counts.get(category,0)/total:.2f}%',f'{counts.get(category,0):,} units',delta_color='off')
        note('Neutral means no observed task-score change after deleting that unit. It does not mean useless or ignored: redundancy, interactions and metric insensitivity can produce zero effects.')
        rs=[r for r in data['correlations'] if r['dataset']==ds]
        st.dataframe(pd.DataFrame(rs)[['dataset','budget','method','rho','ci_low','ci_high','n_units','n_clusters']],hide_index=True,width='stretch')
        st.caption('Post-hoc VAL diagnostic · nominal cluster-bootstrap intervals · unit sizes differ. Use the Evidence Lab scatter to select actual source units.')
        source('units','correlations')
        diagnostic_details(ds)
    with tabs[5]:retrieval_findings(ds,technical or detail=='Technical View')
    with tabs[6]:reader_findings(ds,technical or detail=='Technical View')
    with tabs[7]:complexity_and_protocol(ds,data)
    st.divider()
    matched=[]
    for r in data['directions']:
        if r['method']=='FIXED_SENTENCE':
            for other in ['NONE','RAW_SKIP','QBGF_FULL']:
                a=r.get('VAL_'+other+'_contrast');b=r.get('CONF_'+other+'_contrast')
                if a is not None and b is not None:matched.append((a>0)==(b>0))
    st.metric('Predefined contrast point-estimate directions repeated',f'{sum(matched)} / {len(matched)}')
    st.caption('Descriptive directional consistency, not independent replications. Confirmation is now consumed and cannot be used for new method selection.')
    source('consistency')
    if detail=='Technical View':
        with st.expander('Full result tables',expanded=True):
            st.dataframe(pd.DataFrame(data['validation']),hide_index=True,width='stretch')
            st.dataframe(pd.DataFrame(data['contrasts']),hide_index=True,width='stretch')
    claims()


def claims():
    left,right=st.columns(2)
    with left:
        st.markdown('#### What we established')
        st.markdown('• Evidence benefits the frozen pipeline on reserved cohorts.\n\n• Granularity advantages depend on task and budget.\n\n• Query-conditioned allocation helped on VAL.\n\n• Simple sentence selection is a strong constrained-budget policy.')
    with right:
        st.markdown('#### What we did not establish')
        st.markdown('• Universal QBGF or sentence superiority.\n\n• Semantic similarity being useless, or zero utility from every neutral deletion.\n\n• Latent vectors as the isolated cause of SARA gains.\n\n• SFR as universally best or compression as always beneficial.\n\n• A new state-of-the-art/first knapsack method.\n\n• Fresh QASPER confirmation.')
    source('validation','fusion','contrasts','correlations')


def known_pool(query):
    return recovered_pool(query)


def granularity(query,replays,pool,budget):
    st.subheader('What does evidence granularity mean?')
    note('Ranking asks WHICH evidence is relevant. Granularity asks IN WHAT FORM it reaches the reader. Allocation asks HOW the limited budget is shared.')
    st.markdown('<div class="ladder">FULL PASSAGE → MULTI-SENTENCE → SENTENCE → FINE-GRAINED UNIT<br><span class="note">A spectrum of context per unit—not an ordered quality ranking. Smaller units may let more sources contribute.</span></div>',unsafe_allow_html=True)
    if not pool:
        st.info('Full source decomposition unavailable for this query. Saved numerical attribution remains below. Run the CPU-only recovery tool for supported local VAL sources.');return
    parent=st.selectbox('Same source, different granularity',pool,format_func=lambda p:f"{p['retriever']} rank {int(p['rank'])} · source {p['document_id']} · {p['unit_id'][:10]}")
    hierarchy=query['hierarchy'];children=[u for u in hierarchy if u['passage_id']==parent['unit_id']]
    saved_ids={r['unit_id'] for r in load('units') if r['dataset']==query['dataset'] and r['query_id']==query['query_id']}
    props=[u for u in children if u['unit_type']=='PROPOSITION' and u['unit_id'] in saved_ids]
    options=['Whole Passage','Sentence','Hierarchical','Adaptive / QBGF']+(['Fine-grained source spans'] if props else [])
    view=st.segmented_control('Explore representation',options,default='Sentence')
    if not props:st.caption('Fine-span view unavailable: no saved selected source-span artifact. No result is synthesized.')
    cols=st.columns(3)
    with cols[0]:
        st.markdown('##### Whole passage')
        st.caption(f"{int(parent['token_count'])} source tokens · source {parent['document_id']}")
        with st.container(height=280):st.write(parent['source_text'])
    selected={u['unit_id'] for r in replays if r for u in r['units']}
    selected_parent=parent['unit_id'] in selected
    ss=[u for u in children if u['unit_type']=='SENTENCE']
    with cols[1]:
        retained=[u for u in ss if u['unit_id'] in selected or selected_parent]
        st.markdown('##### Sentences')
        st.caption(f'{len(retained)} exposed sentence spans · {sum(u["token_count"] for u in retained)} standalone source tokens')
        st.caption('Exact-source decomposition · SELECTED means exposed by either displayed replay; the paired cards distinguish the methods.')
        with st.container(height=280):
            for u in ss:unit_card(u,selected=u['unit_id'] in selected or selected_parent)
    with cols[2]:
        st.markdown('##### Fine-grained source spans')
        st.caption(f'{len(props)} saved selected units · {sum(u["token_count"] for u in props)} standalone source tokens')
        with st.container(height=280):
            if props:
                for u in props:unit_card(u,selected=u['unit_id'] in selected)
            else:st.info('No stored selected fine source span for this passage.')
    st.caption('Fine-span availability is checked against saved selections for this query across both budgets. Current SELECTED labels refer only to the displayed method pair and budget, not a new scored context.')
    if view=='Hierarchical':st.info(EXPLANATIONS['H3'])
    if view=='Adaptive / QBGF':
        st.info(EXPLANATIONS['QBGF_FULL'])
        st.caption('Supported actions: DROP · SENT_1 · SENT_2 · SENT_4 · SENT_8 · FULL. Original per-passage action/utility traces were removed in cleanup; this view does not infer them from sentence counts.')
    st.caption('Source token counts are standalone counts. The replay budget meter uses the saved serialized evidence cost, including its block boundaries. These views do not recompute historical selection.')
    source('units','profiles','code')
    with st.expander('The granularity trade-off'):
        st.write('Passages preserve surrounding discourse and co-reference, but fewer may fit. Sentences and rule-derived clauses can expose more sources but lose relationships. QBGF aims to choose fidelity per passage; the mixed-granularity benefit was not consistent in VAL.')
        source('fusion')


def answer_panel(query,replay,method,budget,key):
    replay={**replay,"result_type":"FROZEN_RESEARCH"} if replay is not None else None
    st.caption("FROZEN RESEARCH RESULT")
    st.markdown('#### '+LABELS.get(method,method))
    st.caption(method+' · '+str(budget)+' evidence-token limit')
    explanation(method)
    if replay is None:
        st.info('No compatible saved result for this query × method × budget. The demo does not generate a substitute.');return
    p=replay['profile'];budget_meter(p['evidence_tokens'],budget)
    represented={u['source']['document_id'] for u in replay['units'] if u.get('source')}
    c=st.columns(3);c[0].metric('Selected units',int(p['units']));c[1].metric('Sources verified',len(represented));c[2].metric('Correctness' if query['dataset']=='SciFact' else 'Answer F1',f"{p['full_score']:.3f}")
    st.caption('Prompt / generated tokens: unavailable in the retained per-query attribution export. No estimates are substituted.')
    st.markdown('**Saved answer**')
    if replay['answer'] is not None:st.write(replay['answer'])
    else:st.info('Original answer text unavailable.')
    st.caption(replay['answer_origin'])
    st.caption('Per-claim correctness is not cohort macro-F1.' if query['dataset']=='SciFact' else 'Per-query answer F1. Exact match was not retained in this export.')
    with st.expander('Selected evidence & observed utility',expanded=not presentation):
        for u in replay['units']:unit_card(u)
    source('profiles','units','qualitative')


def lab():
    from demo.curated.publication import load_bundle
    try:
        complete_pack=load_bundle()
    except (ValueError,FileNotFoundError,KeyError) as exc:
        st.error('Curated demo collection failed integrity validation: '+str(exc));complete_pack=None
    if complete_pack:
        collection=st.radio('Replay collection',['Curated examples','Historical exports'],horizontal=True) if technical else 'Curated examples'
        if collection=='Curated examples':
            from demo.curated.ui import render
            render(complete_pack,technical=technical);return
    else:
        st.info('Complete replay collection is not built yet. Run demo/notebooks/BUILD_CURATED_REPLAYS.ipynb to prepare every question × retriever × representation × budget, then publish. Historical exports below may have missing answers.')
    st.subheader('Historical Evidence Replay')
    st.caption('EXPERIMENTAL REPLAY · saved VAL outcomes · instant, model-free exploration')
    tour=st.toggle('Start 90-second tour',value=False)
    if tour:
        step=st.select_slider('Guided presentation',options=['1 · Retrieval','2 · Sources','3 · Granularity','4 · Tight budget','5 · Larger budget','6 · Adaptive choices','7 · Utility','8 · Findings'])
        guidance={'1 · Retrieval':'Compare the retained BM25/SFR pools and note unavailable MiniLM traces; do not infer absence from a partial pool.','2 · Sources':'Open the known retrieved sources. Retrieved does not mean exposed.','3 · Granularity':'Open one source and compare its passage and sentence views.','4 · Tight budget':'Set 256 and compare available evidence policies.','5 · Larger budget':'Switch to 1024; inspect the actual change, without assuming every example improves.','6 · Adaptive choices':'Choose QBGF and explain the supported per-passage fidelity options.','7 · Utility':'Click a scatter point and inspect the matching source unit.','8 · Findings':'Open Research Findings to distinguish VAL mechanisms from confirmation.'}
        note(guidance[step])
        if st.session_state.get('last_tour_step')!=step:
            if step=='4 · Tight budget':st.session_state['lab_budget']=256
            elif step=='5 · Larger budget':st.session_state['lab_budget']=1024
            if step=='2 · Sources':st.session_state['lab_retriever']='SFR'
            st.session_state['last_tour_step']=step
        if step=='8 · Findings':st.button('Open Research Findings',on_click=lambda:st.session_state.__setitem__('workspace','Research Findings'))
    controls=st.columns([1,1,2])
    with controls[0]:dataset=st.selectbox('Dataset',DATASETS,index=2)
    ids=curated_ids(dataset)
    with controls[1]:budget=st.segmented_control('Evidence-token budget',BUDGETS,default=256,key='lab_budget') or 256
    with controls[2]:
        query_id=st.selectbox('Curated validation example',ids,format_func=lambda q:q+' · '+index.get(dataset+':'+q,{}).get('question','Saved numerical profile')[:75])
    with st.expander('Search any saved attribution query ID'):
        entered=st.text_input('Exact query ID',placeholder='Uses saved VAL attribution membership only')
    if entered:query_id=entered.strip()
    try:query=get_query(dataset,query_id,index)
    except DataError as e:st.warning(str(e));return
    st.caption('Outcome-curated examples for explanation, not a representative sample or a new evaluation. Confirmation questions and labels are not used in this lab.')
    st.markdown('### '+(query['question'] or 'Question text unavailable in the retained replay artifacts'))
    if technical:st.caption(f"{dataset} · VAL · query {query_id}")
    task_card(dataset)
    with st.expander('Gold reference · evaluation only'):
        if query['gold']:st.write(query['gold'].get('gold_label') or query['gold'].get('gold_answer'))
        else:st.info('Gold reference unavailable. No synthetic reference is provided.')
    retriever=retriever_explorer(query)
    query=retriever_scope(query,retriever)
    if not query['profiles']:
        st.info('Retrieval-only exploration: no compatible downstream VAL replay for this query and retriever. Aggregate matched comparisons are in Research Findings. No answer is substituted.')
        return
    available=sorted({r['method'] for r in query['profiles'] if r['budget']==budget})
    method_options=list(METHODS)+[m for m in available if m not in METHODS]
    left_default='QBGF_FULL' if 'QBGF_FULL' in available else available[0]
    right_default=next((m for m in available if m!=left_default),left_default)
    context=(dataset,query_id,budget,retriever)
    if (st.session_state.get('comparison_context')!=context
            or st.session_state.get('left_method') not in method_options
            or st.session_state.get('right_method') not in method_options):
        st.session_state['left_method']=left_default;st.session_state['right_method']=right_default
        st.session_state['comparison_context']=context
    controls=st.columns(2)
    with controls[0]:left=st.selectbox('Left method',method_options,index=None,format_func=lambda m:m+(' · no saved row' if m not in available else ''),key='left_method') or left_default
    with controls[1]:right=st.selectbox('Right method',method_options,index=None,format_func=lambda m:m+(' · no saved row' if m not in available else ''),key='right_method') or right_default
    lr=method_replay(query,left,budget);rr=method_replay(query,right,budget)
    pool=[p for p in known_pool(query) if p['retriever']==retriever]
    with st.expander('Before selection · retrieved candidates',expanded=False):
        sfr=pool;st.markdown(f'**Frozen {retriever} width: 10 · recoverable parent sources: {len(sfr)}**')
        complete={int(p['rank']) for p in sfr}==set(range(1,11))
        st.caption('All ten parent ranks are recoverable from saved metadata for this query.' if complete else 'Only part of the original pool is recoverable for this query; missing ranks/scores remain unavailable.')
        st.caption('Entries are source-verified parents of saved selected units. No retrieval rerun. BM25 entries are labelled separately; original decision traces remain unavailable.')
        for p in pool:
            with st.expander(f"{p['retriever']} · rank {int(p['rank'])} · {p['document_id']} · {int(p['token_count'])} source tokens"):
                st.caption(f"Passage {p['unit_id']} · saved relevance {p['score']:.4f}");st.write(p['source_text'])
        st.info('Retrieved candidates are not necessarily all sent to the reader.')
        source('units')
    c=st.columns(2)
    with c[0]:answer_panel(query,lr,left,budget,'left')
    with c[1]:answer_panel(query,rr,right,budget,'right')
    if lr or rr:
        with st.expander('Per-source representation · observed saved selection'):
            allocation=[]
            for method,replay in [(left,lr),(right,rr)]:
                if not replay:continue
                for p in pool:
                    expected_retriever='BM25' if method=='BM25_PLAIN_RAG' else 'SFR'
                    if p['retriever']!=expected_retriever:continue
                    group=[u for u in replay['units'] if u.get('source') and u['source']['passage_id']==p['unit_id']]
                    kinds={u['granularity'] for u in group}
                    representation='Not exposed' if not group else 'Whole passage' if kinds=={'PASSAGE'} else f'{len(group)} '+('/'.join(sorted(kinds))).lower()+' units'
                    allocation.append({'Method':method,'Retrieved rank':int(p['rank']),'Source':p['document_id'],'Representation':representation,'Standalone source tokens':sum(u['tokens'] for u in group),'Optimizer utility':'Not retained'})
            st.dataframe(pd.DataFrame(allocation),hide_index=True,width='stretch')
            st.caption('This table reports saved unit membership. It does not invent missing SENT_k option names or objective values. Source-token sums exclude serialization boundaries.')
            source('units','profiles')
    st.divider();granularity(query,[lr,rr],pool,budget)
    packing_panel(query,pool,[lr,rr],budget)
    st.subheader('Budget changes what can be exposed')
    records=[]
    for r in query['profiles']:
        units=method_replay(query,r['method'],r['budget'])['units']
        parents={u['source']['passage_id'] for u in units if u.get('source')};docs={u['source']['document_id'] for u in units if u.get('source')}
        records.append({'Method':LABELS.get(r['method'],r['method']),'Budget':int(r['budget']),'Evidence tokens':int(r['evidence_tokens']),'Selected units':int(r['units']),'Verified passages represented':len(parents),'Verified documents represented':len(docs),'Correctness' if dataset=='SciFact' else 'Answer F1':r['full_score']})
    st.dataframe(pd.DataFrame(records),hide_index=True,width='stretch')
    st.caption('Actual saved query results—not a guarantee that larger budgets or finer units always improve an answer.')
    source('profiles','units')
    st.subheader('What actually mattered?')
    chosen=st.radio('Inspect utility for',[left,right] if left!=right else [left],horizontal=True,key='utility_method')
    replay=lr if chosen==left else rr
    if replay and replay['units']:
        st.caption('Importance = original task score − score after deleting the unit. Helpful: deletion reduced score. Harmful: deletion improved score. Neutral: no observed metric change.')
        event=st.plotly_chart(attribution(replay['units']),width='stretch',key='utility_scatter',on_select='rerun',selection_mode=('points',))
        points=event.selection.points if event else []
        selected_id=points[0].get('customdata',[None])[0] if points else None
        options=[u['unit_id'] for u in replay['units']]
        if selected_id not in options:selected_id=options[0]
        if selected_id and points and st.session_state.get('last_plot_unit')!=selected_id:
            st.session_state['focused_unit']=selected_id;st.session_state['last_plot_unit']=selected_id
        if st.session_state.get('focused_unit') not in options:st.session_state['focused_unit']=selected_id
        focused=st.selectbox('Inspect a unit · click the chart or select its ID',options,index=None,key='focused_unit')
        unit_card(next(u for u in replay['units'] if u['unit_id']==focused),focus=True)
        note('A zero leave-one-out effect does not prove the reader ignored the evidence. Redundancy, interactions or metric insensitivity can produce a zero effect.')
        if chosen=='BM25_PLAIN_RAG':st.caption('This lexical BM25 source uses BM25 relevance scores, not semantic cosine. It is not merged with the SFR scale.')
        source('units','profiles')
    saved_cases(dataset)


def journey():
    st.subheader('From adaptive context size to evidence allocation')
    with st.expander('Where the project started · SARA and text/latent evidence'):
        st.write('Early work asked whether adaptive text/latent evidence budgets could improve SARA. The later study separated retrieval, granularity, allocation and reader utilization.')
        sara_panel('journey_sara')
    steps=[('W10 · Diagnose','Stronger retrieval helped; reader adaptation and evidence interfaces also mattered.','w10'),
           ('W11 · Broaden','A three-dataset pilot revealed task- and budget-dependent granularity effects.','w11'),
           ('W12 C · Compare','QBGF did not establish a robust advantage over strong simple controls.','validation'),
           ('W12 D · Isolate','Removing query conditioning lowered VAL quality; FULL and SENTENCE_ONLY were often nearly identical.','fusion'),
           ('W12 H/I · Freeze','Retain SFR → FIXED_SENTENCE → Qwen BASE_FP16; no LoRA.','freeze'),
           ('W12 K/L · Explain','Relevance showed weak and context-dependent alignment with input-deletion utility.','correlations'),
           ('W12 M · Confirm','Frozen comparisons broadly repeated their VAL directions on the reserved two-task cohort.','contrasts')]
    for title,text,src in steps:
        with st.container(border=True):st.markdown('#### '+title);st.write(text);source(src)
    st.caption('W10/W11 narrative is orientation, not a new numerical result. The final main tables preserve exact scope and uncertainty.')
    claims()

if mode=='LiveDemo':
    from demo.live.ui import render as live_demo
    live_demo(technical=technical,presentation=presentation)
elif mode=='Research Findings':research_findings()
elif mode=='Research Journey':journey()
else:lab()
st.divider()
st.caption('CS787 · Evidence Allocation in RAG · Research results remain read only · Custom live runs are separate demonstrations')

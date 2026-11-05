"""Presentation UI for actual operational stages, never hidden model reasoning."""
from html import escape
import threading
import pandas as pd
import streamlit as st
from demo.components import unit_card,budget_meter
from .engine import METHODS,STAGES,parse_passages,validate,input_key,run,MAX_PASSAGES
from .resources import LiveResources,inventory
from .segmentation import MODES,ADAPTED,FROZEN
from .protocol import NATIVE,CONTROLLED,SCORING_MODES,assert_controlled_pair,validity,archive_obsolete
from .preset import load_preset,is_preset,passage_status
from .history import remember,rows,differences
from .reference import evaluate as evaluate_reference,metrics_text

METHOD_LABELS={'NONE':'No Evidence','RAW_SKIP':'Whole passages · RAW_SKIP','FIXED_SENTENCE':'Query-selected evidence · FIXED_SENTENCE','QBGF_FULL':'Adaptive fidelity · QBGF'}
WHY={
 'NONE':'No retrieval or evidence allocation occurs. The reader answers from the query alone.',
 'RAW_SKIP':'Entire passages are preserved, but a passage can only be included if it fits within the remaining evidence budget. Oversized passages are skipped and later ones can still be included.',
 'FIXED_SENTENCE':'Retrieved passages are decomposed into source-linked evidence units using the selected segmentation policy. Fitting relevant units are selected and restored to source order. Their badges distinguish sentences, headings, lists and other source segments.',
 'QBGF_FULL':'Each retrieved passage can be dropped, represented by selected evidence units, or preserved in full. These fidelity choices are optimized jointly under one shared evidence-token budget, rather than simply reranking passages.'}
SAMPLE=[
 'The Helios laboratory compared three temperature sensors. Sensor A recorded measurements once every second. Its battery lasted twelve hours.',
 'Sensor B recorded a measurement every ten seconds. It had a battery life of forty-eight hours. The team selected it for remote deployments where replacing batteries was difficult.',
 'Sensor C sampled twice per second. It was used for short experiments where rapid changes mattered more than battery life.',
 'The remote field campaign lasted thirty-six hours. Its primary requirement was unattended operation for the entire campaign.',
 'The laboratory report also measured enclosure mass and calibration time. These were secondary requirements for the remote campaign.']

@st.cache_resource(show_spinner=False)
def resources():return LiveResources()
@st.cache_data(ttl=5,show_spinner=False)
def available():return inventory()

def pipeline_html(states):
    icons={'pending':'○','processing':'◌','completed':'✓','failed':'!','skipped':'–'}
    return '<div class="live-pipeline">'+''.join('<div class="live-stage '+states[s]['state']+'"><b>'+icons[states[s]['state']]+'</b><span>'+s+'</span><small>'+escape(states[s]['state'])+'</small></div>' for s in STAGES)+'</div>'

def candidate_table(r):
    selected={u['passage_id'] for u in r['selection']['units']}
    return pd.DataFrame([dict(Rank=x['rank'],Passage=x['unit']['document_id'],Relevance=round(x['score'],5),Tokens=x['unit']['token_count'],Status='Exposed' if x['unit']['unit_id'] in selected else 'Candidate — not exposed' if x['in_pool'] else 'Outside top-10 pool') for x in r['ranking']])

def evidence_card(r,technical=False):
    st.caption('LIVE CUSTOM RUN')
    st.markdown('#### Model Answer')
    if r['answer'] is None:st.info('Reader generation unavailable in this environment. The evidence pipeline completed and the exact context is available below.')
    else:st.write(r['answer'])
    if is_preset(r['query'],r['passages']):
        st.caption('Preset Reference Answer comparison · user-supplied demonstration reference, not official gold.')
        st.caption(metrics_text(r))
        with st.expander('Preset Reference Answer and required-facts checklist'):
            st.write(load_preset()['reference']['answer'])
            metrics=evaluate_reference(r)
            if metrics is not None:st.dataframe(pd.DataFrame(metrics['checklist']),hide_index=True,width='stretch')
            else:st.write([f['text'] for f in load_preset()['reference']['required_facts']])
            st.caption('F1, precision and recall measure normalized token overlap; EM is normalized exact match. Fact cues are rule-based mentions, not semantic correctness or contradiction detection. Paraphrases may be missed.')
    else:st.caption('No gold-reference evaluation is available for custom LiveDemo queries.')
    if r['answer'] is not None:st.caption('Custom answers are not automatically verified for factual correctness or complete support. Inspect the exact selected evidence to assess grounding.')
    st.markdown('#### Evidence Budget');budget_meter(r['selection']['evidence_tokens'],r['budget'])
    c=st.columns(4)
    for col,label,value in zip(c,['Source tokens','Evidence units','Prompt tokens','Generated tokens'],[r['source_tokens'],len(r['selection']['units']),r['prompt_tokens'],r['generated_tokens'] if r['generated_tokens'] is not None else '—']):col.metric(label,value)
    st.markdown('#### Exact Evidence Sent to the Reader' if r['answer'] is not None else '#### Exact Evidence Prepared for the Reader')
    if r['method']=='NONE':st.info('No external evidence was supplied. The reader is configured to use only the query and its pretrained knowledge.')
    elif not r['selection']['units']:st.info('No evidence unit fit this allocation; the prepared reader context is empty.')
    else:
        st.code(r['selection']['text'],language=None)
        for u in r['selection']['units']:unit_card(u)
    if technical:
        with st.expander('Technical / provenance'):
            st.json(dict(reader=r['reader'],device=r.get('device'),reader_error=r['reader_error'],durations_seconds=r['durations'],measured_run_seconds=r['measured_run_seconds'],query_tokens=r['query_tokens'],retriever=r['retriever'],evidence_ids=r['selection']['evidence_ids']))
            st.caption('Timings describe this interactive run, including loading and interface updates; they are not benchmark latency claims.')

def show_result(r,technical=False):
    st.subheader('User Query');st.write(r['query'])
    st.caption('LIVE CUSTOM RUN · Segmentation: '+('Citation/Markdown-aware Live' if r.get('segmentation')==ADAPTED else 'Frozen research splitter'))
    st.caption('CONTROLLED RETRIEVER EFFECT · Only candidate retrieval differs.' if r.get('scoring_mode')==CONTROLLED else 'Pipeline-native comparison · Retriever and downstream relevance scoring may both differ.')
    if r.get('segmentation')==FROZEN:st.warning('Historical fallback mode can merge citation-heavy text. These are not guaranteed grammatical sentence units.')
    if technical:
        with st.expander('Sentence segmentation audit · raw passages and every unit'):
            for i,p in enumerate(r['passages'],1):st.markdown(f'**Raw passage P{i}**');st.code(p,language=None)
            st.dataframe(pd.DataFrame(r.get('sentence_inventory',[])),hide_index=True,width='stretch')
            import json
            st.download_button('Download segmentation audit JSON',json.dumps(dict(passages=r['passages'],splitter=r.get('segmentation'),sentences=r.get('sentence_inventory',[])),indent=2,ensure_ascii=False),file_name='live_sentence_audit.json',mime='application/json',key='segmentation_export_'+r['input_key']+'_'+r['method'])
    st.caption(f"{METHOD_LABELS[r['method']]} · {r['retriever'] if r['method']!='NONE' else 'Retrieval skipped'} · {r['budget']} evidence-token budget · Qwen3-8B")
    if r['ranking']:
        st.subheader('Retrieved Candidate Pool');st.dataframe(candidate_table(r),hide_index=True,width='stretch')
        for x in r['ranking']:
            with st.expander(f"#{x['rank']} · {x['unit']['document_id']} · {'candidate' if x['in_pool'] else 'outside top-10'}"):
                st.caption(f"Relevance {x['score']:.4f} · {x['unit']['token_count']} source tokens");st.write(x['unit']['source_text'])
    if is_preset(r['query'],r['passages']):
        st.markdown('#### Budget-sensitive multi-aspect question')
        st.caption('Static diagnostic scenario, not factual ground truth or a tuning set. Retrieval ≠ selection ≠ allocation ≠ reader utilization.')
        st.dataframe(pd.DataFrame(passage_status(r)),hide_index=True,width='stretch')
        with st.expander('Inspect P7 workload and P9 abandoned-format evidence',expanded=True):
            for pid in ('P7','P9'):
                st.markdown('**'+pid+'**');st.code(r['passages'][int(pid[1:])-1],language=None)
                chosen=[u for u in r['selection']['units'] if u['document_id']==pid]
                st.caption('Selected representation:')
                if not chosen:st.write('No evidence from this passage reached the reader.')
                for u in chosen:unit_card(u)
    evidence_card(r,technical)
    if r['allocation_trace']:
        with st.expander('How the budget was allocated',expanded=r['method'] in ('RAW_SKIP','QBGF_FULL')):
            st.dataframe(pd.DataFrame(r['allocation_trace']),hide_index=True,width='stretch')
            if r.get('qbgf_options'):
                st.markdown('**All deterministic fidelity options**')
                st.dataframe(pd.DataFrame([{k:v for k,v in o.items() if k not in ('evidence_text','unit_types')} for o in r['qbgf_options']]),hide_index=True,width='stretch')
                for o in r['qbgf_options']:
                    if o['selected']:
                        st.markdown(f"**{o['passage_id']} → {o['label']} · cost {o['cost']} · utility {o['utility']:.5f}**")
                        st.caption('Unit types: '+', '.join(o['unit_types']));st.code(o['evidence_text'] or '(dropped)',language=None)
            if r['method']=='QBGF_FULL':st.caption('Actual global optimization over DROP, 1/2/4/8 selected evidence units and FULL PASSAGE. Internal SENT_k IDs are compatibility names, not grammar guarantees. Utility is the relevance surrogate, not answer quality.')
    if r['pool']:
        st.subheader('Same source, different granularity')
        p=st.selectbox('Inspect source passage',r['pool'],format_func=lambda u:u['document_id'],key='live_source_'+r['input_key']+'_'+r['method'])
        children=[u for u in r['sentences'] if u['passage_id']==p['unit_id']];chosen=[u for u in r['selection']['units'] if u['passage_id']==p['unit_id']];ids={u['unit_id'] for u in chosen}
        c=st.columns(3)
        with c[0]:st.markdown('##### Full passage');unit_card(p,selected=p['unit_id'] in ids)
        with c[1]:
            st.markdown('##### Sentences / source segments')
            for u in children:
                unit_card(dict(u,relevance=r['relevance'].get(u['unit_id'])),selected=u['unit_id'] in ids or p['unit_id'] in ids)
        with c[2]:
            st.markdown('##### Selected representation')
            for u in chosen:unit_card(u)
            if not chosen:st.info('This source was not exposed to the reader.')
    st.subheader('What happened in this run?')
    for line in r['explanation']:st.write(line)
    with st.expander('Why does this method behave this way?',expanded=True):st.write(WHY[r['method']])
    with st.expander('Inspect query coverage · descriptive only'):
        st.write(r['query']);st.code(r['selection']['text'] or '(no external evidence)',language=None)
        st.caption('Inspect which selected excerpts address each part of your question. No automatic coverage score, correctness metric or selection signal is computed.')


def render(technical=False,presentation=True):
    st.subheader('LiveDemo')
    st.write('Bring your own passages and query. Follow the evidence from retrieval to the reader’s answer.')
    pool=resources();inv=available()
    device=st.selectbox('Live model device',['cpu','cuda'],format_func=lambda d:'CPU · FP16 · sequential models' if d=='cpu' else 'GPU · FP16 · sequential models',key='live_device')
    if hasattr(pool,'set_device'):pool.set_device(device)
    if not technical and st.session_state.get('live_segmentation')==FROZEN:st.session_state['live_segmentation']=ADAPTED
    segmentation=st.selectbox('Source segmentation',MODES if technical else (ADAPTED,),format_func=lambda m:'Citation/Markdown-aware Live segmentation' if m==ADAPTED else 'Frozen research splitter · exact historical behavior',key='live_segmentation')
    st.caption('Recommended for pasted articles, Wikipedia text, citations, headings and lists. Uses the frozen splitter plus a source-preserving boundary adapter.')
    if segmentation==FROZEN:st.warning('This reproduces the historical fallback splitter exactly. Citation-heavy pasted text may merge multiple grammatical sentences because the original benchmark pipeline was not designed specifically for raw Wikipedia/Markdown input.')
    st.caption('Citation/Markdown mode adds explicit live-only boundaries. Frozen mode preserves research behavior, including its citation limitations.')
    enable=st.toggle('Enable Live Models',value=False,key='live_models_enabled',help='Opt in to local SFR/Qwen inference. CPU FP16 runs SFR then Qwen sequentially; BM25 context-only exploration needs no model weights. Nothing is downloaded.')
    if not technical and st.session_state.get('live_scoring_mode')==NATIVE:st.session_state['live_scoring_mode']=CONTROLLED
    scoring_mode=st.selectbox('Comparison protocol',(CONTROLLED,NATIVE) if technical else (CONTROLLED,),format_func=lambda m:'Controlled retriever-only comparison' if m==CONTROLLED else 'Pipeline-native comparison',key='live_scoring_mode')
    if scoring_mode==CONTROLLED:st.info('CONTROLLED RETRIEVER EFFECT · Only candidate retrieval differs. Both pools use the same SFR passage/unit scorer and frozen downstream settings.')
    else:st.caption('Retriever and its associated downstream relevance scoring may both differ. This compares complete pipeline variants, not retrieval alone. BM25-native QBGF is disabled because of score-scale incompatibility.')
    tools=st.columns([4,1])
    with tools[0]:st.caption('Models load only when you run the pipeline. SFR and Qwen take turns on the selected device. CPU mode unloads both after each answer. Curated Demo remains model-free.')
    with tools[1]:
        if st.button('Unload live models'):
            with pool.lock:pool.release_gpu()
            st.success('Live models unloaded.')
    if 'live_passage_ids' not in st.session_state:st.session_state['live_passage_ids']=[0,1,2];st.session_state['live_next_id']=3
    states={s:dict(state='pending',detail='') for s in STAGES}
    with st.container(key='live_progress'):
        progress=st.empty();progress.markdown(pipeline_html(states),unsafe_allow_html=True)
        activity=st.empty()
    left,right=st.columns([1.25,1])
    with left:
        st.markdown('### Evidence Passages')
        if st.button('Budget-sensitive multi-aspect question',key='live_world_cup_preset'):
            preset=load_preset();values=preset['passages']
            st.session_state['live_passage_ids']=list(range(len(values)));st.session_state['live_next_id']=len(values)
            for i,text in enumerate(values):st.session_state[f'live_passage_{i}']=text
            st.session_state['live_query']=preset['query'];st.rerun()
        if st.button('Try a sample',key='live_sample'):
            st.session_state['live_passage_ids']=list(range(5));st.session_state['live_next_id']=5
            for i,p in enumerate(SAMPLE):st.session_state[f'live_passage_{i}']=p
            st.session_state['live_query']='Which sensor was selected for remote deployment, and why?';st.rerun()
        for pos,pid in enumerate(list(st.session_state['live_passage_ids'])):
            c=st.columns([5,1])
            with c[0]:st.text_area(f'Passage {pos+1}',key=f'live_passage_{pid}',height=105,max_chars=8000)
            with c[1]:
                if len(st.session_state['live_passage_ids'])>1 and st.button('Remove',key=f'live_remove_{pid}'):
                    st.session_state['live_passage_ids'].remove(pid);st.rerun()
        if st.button('+ Add Passage',disabled=len(st.session_state['live_passage_ids'])>=MAX_PASSAGES):
            pid=st.session_state['live_next_id'];st.session_state['live_next_id']=pid+1;st.session_state['live_passage_ids'].append(pid);st.rerun()
        passages=[st.session_state.get(f'live_passage_{i}','') for i in st.session_state['live_passage_ids']]
        st.caption(f"{sum(bool(p.strip()) for p in passages)} passages loaded · approximately {round(sum(len(p.split()) for p in passages)*1.3)} tokens (word-based estimate)")
        with st.expander('Paste multiple passages at once'):
            st.text_area('Separate passages with a line containing ---',key='live_bulk',height=160,max_chars=30000)
            if st.button('Import passages'):
                values=parse_passages(st.session_state.get('live_bulk',''))
                try:
                    validate('Import',values,'RAW_SKIP',256,'BM25')
                    start=st.session_state['live_next_id'];st.session_state['live_passage_ids']=list(range(start,start+len(values)));st.session_state['live_next_id']=start+len(values)
                    for i,value in zip(st.session_state['live_passage_ids'],values):st.session_state[f'live_passage_{i}']=value
                    st.rerun()
                except ValueError as exc:st.error(str(exc))
    with right:
        st.markdown('### Query');query=st.text_area('Your query',key='live_query',height=125,max_chars=2000)
        budget=st.segmented_control('Evidence Budget',[256,1024],default=256,key='live_budget') or 256
        method=st.selectbox('Method',METHODS,index=2,format_func=lambda m:METHOD_LABELS[m],key='live_method')
        capacity=pool.available_memory_mib() if enable and device=='cuda' else inv['gpu_free_mib']
        sfr_usable=enable and inv['sfr_local'] and (device=='cpu' or (capacity is not None and (capacity>=16000 or pool.encoder is not None or pool.reader is not None)))
        opts=['BM25']+(['SFR'] if sfr_usable else [])
        if st.session_state.get('live_retriever') not in opts:st.session_state['live_retriever']='BM25'
        retriever=st.selectbox('Retriever',opts,key='live_retriever')
        if scoring_mode==CONTROLLED:
            st.caption('RAW_SKIP: retrieval rank determines whole-passage packing order. FIXED_SENTENCE/QBGF: the common SFR scorer controls downstream allocation; the retriever primarily changes which passages enter the pool, not the unit relevance ranking.')
        else:st.caption('Advanced native mode changes lexical/semantic downstream scoring along with retrieval; it is not a pure retriever comparison.')
        if not enable:st.info('Context-only BM25 works for NONE, RAW_SKIP and FIXED_SENTENCE. QBGF and controlled comparisons require the local SFR scorer; enable live models to request scoring and an answer.')
        elif not inv['qwen_local']:st.info('Qwen is unavailable locally. You can still inspect evidence processing.')
        blocked=method=='QBGF_FULL' and retriever=='BM25' and scoring_mode==NATIVE
        if blocked:st.warning('For QBGF with BM25 candidates, select Controlled retriever-only comparison and enable local SFR scoring.')
        if scoring_mode==CONTROLLED and not enable:st.info('Controlled mode needs local SFR scoring; enable Live Models before running.')
        pressed=st.button('Run Evidence Pipeline',type='primary',width='stretch',disabled=blocked or (scoring_mode==CONTROLLED and not enable and method!='NONE'))
    details=st.container();slots={}
    with details:
        for stage in STAGES:slots[stage]=st.empty()
    cache=st.session_state.setdefault('live_prepared',{});results=st.session_state.setdefault('live_results',{})
    obsolete=st.session_state.setdefault('live_obsolete',{});archive_obsolete(results,obsolete)
    history=st.session_state.setdefault('live_history',[])
    for saved in results.values():remember(history,saved)
    if obsolete:
        with st.expander('Previous live runs · obsolete protocol metadata'):
            st.warning('PRE-FIX LIVE DEMO · OBSOLETE FOR GRANULARITY INTERPRETATION. Old contexts are retained for inspection but cannot be reused in corrected comparisons. RAW_SKIP is not invalid solely because sentence segmentation changed.')
            st.json([dict(method=r.get('method'),segmentation=r.get('segmentation'),note=r.get('validity_note')) for r in obsolete.values()])
    def notify(stage,state,detail,states):
        progress.markdown(pipeline_html(states),unsafe_allow_html=True)
        activity.caption(stage+' · '+detail)
        with slots[stage].container():
            st.caption(stage+' · '+detail)
            preview=states[stage].get('preview')
            if preview is not None:
                with st.expander(stage+' details',expanded=not presentation):
                    if isinstance(preview,list):st.dataframe(pd.DataFrame(preview),hide_index=True,width='stretch')
                    elif stage=='Allocation':budget_meter(preview['evidence_tokens'],preview['budget']);st.dataframe(pd.DataFrame(preview['decisions']),hide_index=True,width='stretch')
                    else:st.json(preview)
    def execute(q,ps,m,b,rt,reader=True,score_mode=None,segment_mode=None):
        sm=score_mode or scoring_mode;seg=segment_mode or segmentation
        k=(input_key(q.strip(),[p.strip() for p in ps],b,rt,seg,sm),m,device)
        old=results.get(k)
        if reader and old is not None and validity(old) is None and (old['answer'] is not None or not enable):
            remember(history,old)
            st.info('Reused this session’s completed result; no model call was made.')
            return old
        with st.spinner('Running evidence operations…'):
            with pool.lock:
                if hasattr(pool,'set_device'):pool.set_device(device)
                value=run(q,ps,m,b,rt,pool,enable,notify,cache,generate_answer=reader,segmentation=seg,scoring_mode=sm)
        if reader:
            results[k]=value
            remember(history,value)
        if len(results)>20:
            oldest=next(iter(results))
            if oldest!=k:results.pop(oldest)
        return value
    if pressed:
        try:
            validate(query,passages,method,budget,retriever)
            value=execute(query,passages,method,budget,retriever);st.session_state['live_active']=(value['input_key'],method,device);st.session_state.pop('live_compare',None)
        except Exception as exc:st.error(str(exc))
    active=st.session_state.get('live_active');value=results.get(active) if active else None
    if value and validity(value) is None and value.get('device',device)==device:
        if value['input_key']!=input_key(query.strip(),[p.strip() for p in passages],budget,retriever,segmentation,scoring_mode) or value['method']!=method:
            st.caption('Showing the last completed run. Run the pipeline to apply edited inputs.')
        progress.markdown(pipeline_html(value['stages']),unsafe_allow_html=True)
        st.divider();show_result(value,technical)
        st.subheader('Compare with another method')
        st.caption('Compare how different evidence policies change the context supplied to the same reader. The original query, passages, budget and retriever are preserved.')
        other=st.selectbox('Comparison method',[m for m in METHODS if m!=value['method']],format_func=lambda m:METHOD_LABELS[m],key='live_compare_method_'+value['method'])
        if st.button('Run comparison',disabled=other=='QBGF_FULL' and value['retriever']=='BM25' and value['scoring_mode']==NATIVE):
            try:
                second=execute(value['query'],value['passages'],other,value['budget'],value['retriever'],score_mode=value['scoring_mode'],segment_mode=value['segmentation']);st.session_state['live_compare']=(second['input_key'],other,device)
            except Exception as exc:st.error(str(exc))
        compared=results.get(st.session_state.get('live_compare'))
        if compared and compared['input_key']==value['input_key'] and compared.get('device',device)==device:
            c=st.columns(2)
            for col,r in zip(c,[value,compared]):
                with col:st.markdown('### '+METHOD_LABELS[r['method']]);evidence_card(r,technical)
            st.dataframe(pd.DataFrame([dict(Method=METHOD_LABELS[r['method']],Evidence_tokens=r['selection']['evidence_tokens'],Sources_represented=len({u['document_id'] for u in r['selection']['units']}),Evidence_units=len(r['selection']['units']),Prompt_tokens=r['prompt_tokens'],Answer=r['answer'] or 'Reader unavailable') for r in [value,compared]]),hide_index=True,width='stretch')
        if sfr_usable:
            if st.button('Compare Retrievers'):
                try:
                    st.session_state['live_retrieval_compare']=[execute(value['query'],value['passages'],'RAW_SKIP',value['budget'],rt,reader=False,score_mode=CONTROLLED,segment_mode=value['segmentation']) for rt in ('BM25','SFR')]
                except Exception as exc:st.error(str(exc))
            rc=st.session_state.get('live_retrieval_compare')
            if rc and same_custom_inputs(rc[0],value):
                cols=st.columns(2)
                for col,r in zip(cols,rc):
                    with col:st.markdown('#### '+r['retriever']+' ranking');st.dataframe(candidate_table(r),hide_index=True,width='stretch')
            if st.button('Compare Retrievers End-to-End',disabled=value['method']=='NONE'):
                try:
                    pair=[execute(value['query'],value['passages'],value['method'],value['budget'],rt,score_mode=CONTROLLED,segment_mode=value['segmentation']) for rt in ('BM25','SFR')]
                    assert_controlled_pair(*pair);st.session_state['live_e2e']=pair
                except Exception as exc:st.error(str(exc))
            pair=st.session_state.get('live_e2e')
            if pair and same_custom_inputs(pair[0],value) and pair[0]['method']==value['method']:
                assert_controlled_pair(*pair)
                st.subheader('CONTROLLED RETRIEVER EFFECT')
                st.info('Only the candidate retriever differs in this comparison. This view demonstrates how changing candidate retrieval propagates through selection into context and answer; no custom-input winner is scored.')
                with st.expander('Pipeline invariants',expanded=True):
                    st.write('Held constant: source passages, query, segmentation, common SFR downstream scorer, allocation method, evidence budget, Qwen reader, prompt template/parser, decoding and device. Changed: candidate retriever only.')
                    if technical:st.json(pair[0]['invariants'])
                columns=st.columns(2)
                for col,r in zip(columns,pair):
                    with col:
                        st.markdown('### '+r['retriever']);st.write('Top-k IDs: '+', '.join(u['document_id'] for u in r['pool']))
                        st.dataframe(candidate_table(r),hide_index=True,width='stretch');evidence_card(r,technical)
                        st.caption('Unit types: '+', '.join(u.get('display_kind',u['unit_type']) for u in r['selection']['units']))
    render_history(history,technical)


def render_history(history,technical=False):
    """Render saved pairs with the existing evidence cards; no execution path."""
    st.divider();st.subheader('Run History / Comparison Tray')
    st.caption('Last 30 distinct runs in this browser session. Select any two to inspect saved results without new model calls. Context-only runs are marked explicitly.')
    summary=rows(history)
    if not summary:
        st.info('Complete a live pipeline run to start your history.')
        return
    st.dataframe(pd.DataFrame(summary),hide_index=True,width='stretch',row_height=90,
                 column_config={'Answer':st.column_config.TextColumn('Saved answer',width='large'),
                                'Metrics':st.column_config.TextColumn('Reference metrics',width='large')})
    if any(row['Run']==0 for row in summary):
        st.caption('Row 0 is the Preset Reference Answer, not a generated run. Metrics apply only to runs with the exact frozen preset query and passages. Other inputs remain unscored.')
        with st.expander('Preset reference · metric definitions and required facts'):
            st.write(load_preset()['reference']['answer'])
            for fact in load_preset()['reference']['required_facts']:st.write('• '+fact['text'])
            st.caption('P = overlapping tokens / answer tokens; R = overlapping tokens / reference tokens; F1 = 2PR/(P+R); EM = normalized string equality. Normalization uses the project HotpotQA rules. Fact-cue coverage is matched checklist cues / 9; it is a lexical heuristic, not factual verification. Reference text is never supplied to retrieval, allocation or the reader.')
    entries={e['number']:e for e in history if validity(e['result']) is None}
    if len(entries)<2:
        st.caption('Run another method, budget or retriever to compare two saved runs.')
        return
    def label(n):
        r=entries[n]['result']
        return f"Run {n} · {r['retriever']} · {r['method']} · {r['budget']}"
    picked=st.multiselect('Select two saved runs',list(entries),format_func=label,
                          max_selections=2,key='live_history_picked')
    if st.button('Compare Selected Runs',disabled=len(picked)!=2):
        st.session_state['live_history_pair']=tuple(picked)
    pair=st.session_state.get('live_history_pair',())
    if len(pair)!=2 or any(n not in entries for n in pair):return
    a,b=[entries[n]['result'] for n in pair]
    changed=differences(a,b)
    st.caption('Changed: '+(', '.join(changed) if changed else 'no recorded configuration fields')+'.')
    if any(x not in ('Retriever','Method','Budget') for x in changed) or len(changed)>1:
        st.info('These runs differ in multiple settings or input/protocol details. This is a descriptive comparison, not an isolated effect of one variable.')
    st.caption('Compare the saved evidence and answers; no gold-reference winner is inferred.')
    for col,n,r in zip(st.columns(2),pair,[a,b]):
        with col:
            st.markdown('### '+label(n));st.write(r['query'])
            st.caption('Segmentation: '+r['segmentation']+' · Scoring: '+r['scoring_mode'])
            evidence_card(r,technical)
            with st.expander('Retrieved candidate pool'):
                st.dataframe(candidate_table(r),hide_index=True,width='stretch')
                for x in r['ranking']:
                    st.markdown('**'+x['unit']['document_id']+'**');st.write(x['unit']['source_text'])
            if r['allocation_trace']:
                with st.expander('Budget allocation decisions'):
                    st.dataframe(pd.DataFrame(r['allocation_trace']),hide_index=True,width='stretch')


def same_custom_inputs(a,b):
    return all(a.get(k)==b.get(k) for k in ('query','passages','budget','segmentation','device'))

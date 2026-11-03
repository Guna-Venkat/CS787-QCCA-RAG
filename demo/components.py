from html import escape
from pathlib import Path
import streamlit as st
from demo.constants import ROOT,COLORS,SOURCES,EXPLANATIONS
from demo.data_loader import provenance

def source(*names):
    with st.expander('Source · primary artifacts',expanded=False):
        for name in names:
            try:
                info=provenance(name);st.code(info['path'],language=None)
                st.caption('SHA256 '+info['sha256'][:16]+'… · read only')
            except (ValueError,FileNotFoundError):st.caption('Source unavailable: '+SOURCES.get(name,name))

def hero(compact=False):
    if compact:
        st.markdown('<div class="eyebrow">CS787 · Evidence Allocation Study</div><h2>From retrieval relevance to downstream utility.</h2><p class="note">Frozen results, controlled comparisons, and the limits of each claim.</p>',unsafe_allow_html=True)
        return
    st.markdown('''<div class="hero"><div class="eyebrow">CS787 · Evidence Allocation Study</div>
    <h1>The right evidence.<br>At the right level of detail.</h1>
    <p>Budget-aware evidence granularity in retrieval-augmented generation. Does a reader need full passages—or carefully chosen pieces?</p>
    </div><div class="pipeline"><span>Query</span><b>→</b><span>Retriever</span><b>→</b><span>Top-k pool</span><b>→</b><span>Granularity</span><b>→</b><span>Allocation</span><b>→</b><span>Reader</span><b>→</b><span>Answer</span><b>→</b><span>Utility analysis</span></div>''',unsafe_allow_html=True)

def note(text):st.markdown('<div class="callout">'+escape(text)+'</div>',unsafe_allow_html=True)

def unit_card(u,selected=True,focus=False):
    kind=u.get('display_kind') or u.get('granularity') or (u.get('source') or {}).get('unit_type') or u.get('unit_type','SOURCE')
    text=u.get('text') or u.get('source_text') or 'Exact source text unavailable in retained artifacts.'
    if kind=='PROPOSITION':kind='FINE-GRAINED SOURCE SPAN'
    cat=u.get('category');color=COLORS.get(cat,'#94a3b8')
    tag=('<span class="tag" style="color:'+color+'">'+escape(cat)+' · LOO</span>') if cat else ''
    rel=u.get('relevance');imp=u.get('importance');extra=[]
    if rel is not None:extra.append(f'Relevance {rel:.3f}')
    if imp is not None:extra.append(f'Deletion utility {imp:+.3f}')
    if u.get('tokens',u.get('token_count')) is not None:extra.append(f'{int(u.get("tokens",u.get("token_count")))} source tokens')
    st.markdown('<div class="unit'+(' focus' if focus else '')+'"><span class="tag">'+('SELECTED' if selected else 'NOT SELECTED')+'</span> <span class="tag">'+escape(kind)+'</span>'+tag+'<p>'+escape(text)+'</p><small>'+escape(' · '.join(extra))+'</small></div>',unsafe_allow_html=True)

def explanation(method):
    with st.expander('How this method allocates evidence'):
        st.write(EXPLANATIONS.get(method,'See the original registered method definition.'))
        source('code')

def budget_meter(evidence,budget):
    st.progress(min(1.,max(0.,evidence/budget)),text=f'Evidence budget · {int(evidence):,} / {budget:,} tokens')
    st.caption(f'{100*evidence/budget:.1f}% utilized · evidence excludes question and instruction framing')

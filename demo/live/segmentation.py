"""Live boundary adapter; the frozen academic splitter is imported, never edited.

Citation masking is same-length and used only for finding boundaries. All emitted
text is sliced from the original unmodified passage. This is NOT raw frozen parity
when citation/Markdown boundaries intervene; callers must persist the policy ID.
"""
import re
from src.week9.senttopk_e3.sentences import spans as frozen_spans,SPLITTER

FROZEN=SPLITTER
ADAPTED=SPLITTER+'+live-citation-markdown-v1'
DEFAULT=ADAPTED
MODES=(ADAPTED,FROZEN)
CITATION=re.compile(r'\[(?:\d+(?:\s*[,–—-]\s*\d+)*|[A-Z]|citation needed)\]',re.I)
HEADING=re.compile(r'^\s{0,3}#{1,6}\s+\S')
BULLET=re.compile(r'^\s*(?:[-+*]|\d+[.)])\s+\S')


def sentence_spans(text,mode=DEFAULT):
    if mode not in MODES:raise ValueError('Unknown live segmentation policy')
    if mode==FROZEN:return frozen_spans(text)
    citations=list(CITATION.finditer(text));view=list(text)
    for m in citations:view[m.start():m.end()]=' '*(m.end()-m.start())
    view=''.join(view)
    # Only explicit Markdown structures force single-line boundaries. Ordinary
    # wrapped prose retains the frozen rule (a single newline is not a boundary).
    structural={0,len(text)};offset=0
    for line in text.splitlines(keepends=True):
        if HEADING.match(line) or BULLET.match(line):structural.update((offset,offset+len(line)))
        offset+=len(line)
    cuts=sorted(structural);out=[]
    for lo,hi in zip(cuts,cuts[1:]):
        found=[(lo+a,lo+b) for a,b in frozen_spans(view[lo:hi])]
        # Attach numeric citation suffixes to the preceding sentence, whether
        # adjacent or separated by whitespace. Abbreviations still use the frozen rule.
        for a,b in found:
            cursor=b
            while True:
                nxt=next((m for m in citations if m.start()>=cursor and m.end()<=hi and not text[cursor:m.start()].strip()),None)
                if nxt is None:break
                cursor=nxt.end()
            b=cursor
            while a<b and text[a].isspace():a+=1
            while b>a and text[b-1].isspace():b-=1
            if b>a:out.append((a,b))
    # A citation-only paragraph masked out above belongs to no sentence; preserve
    # it as a source segment rather than lose non-whitespace source characters.
    for m in citations:
        if not any(a<=m.start() and m.end()<=b for a,b in out):out.append((m.start(),m.end()))
    out.sort()
    if any(a<prev_b for (_,prev_b),(a,_) in zip(out,out[1:])):raise ValueError('Overlapping live boundaries')
    return out


def documents(passages,mode=DEFAULT):
    return [dict(dataset='LIVE',document_id=f'P{i+1}',text=p,
                 native_sentences=[dict(start_char=a,end_char=b) for a,b in sentence_spans(p,mode)],
                 native_passages=[dict(start_char=0,end_char=len(p))]) for i,p in enumerate(passages)]


def annotate(units,mode):
    for u in units:
        if u['unit_type']=='SENTENCE':
            text=u['source_text'];u['splitter']=mode
            u['boundary_kind']='heading' if HEADING.match(text) else 'list_item' if BULLET.match(text) else 'prose'
            count=len(sentence_spans(text,ADAPTED))
            punctuated=bool(re.search(r'[.!?]["\'”’\)\]]*$',CITATION.sub('',text).rstrip()))
            u['display_kind']='MULTI-SENTENCE SEGMENT' if count>1 else 'HEADING SEGMENT' if u['boundary_kind']=='heading' else 'LIST SEGMENT' if u['boundary_kind']=='list_item' else 'SOURCE SEGMENT' if '\n' in text or not punctuated else 'SENTENCE'
            u['frozen_subsegment_count']=len(frozen_spans(text))
    return units


def inventory(passages,units,tokenizer,mode):
    result=[]
    for u in units:
        if u['unit_type']!='SENTENCE':continue
        p=passages[int(u['document_id'][1:])-1]
        if p[u['start_char']:u['end_char']]!=u['source_text']:raise ValueError('Live sentence/source mismatch')
        expected=sentence_spans(p,mode)
        if (u['start_char'],u['end_char']) not in expected:raise ValueError('Live sentence is not an exact policy span')
        result.append(dict(passage_id=u['document_id'],parent_passage_id=u['passage_id'],sentence_id=u['sentence_id'],raw_text=u['source_text'],start_char=u['start_char'],end_char=u['end_char'],token_count=len(tokenizer.encode(u['source_text'],add_special_tokens=False)),splitter=mode,boundary_kind=u.get('boundary_kind'),frozen_subsegment_count=len(frozen_spans(u['source_text'])),adapted_subsegment_count=len(sentence_spans(u['source_text'],ADAPTED))))
    return result

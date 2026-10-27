"""Exact source spans; no generated propositions and no label inputs."""
from src.week9.senttopk_e3.sentences import spans as sentence_spans, SPLITTER
from .propositions import spans as proposition_spans, METHOD
from .cache import digest


def unit(doc, kind, a, b, parent, order, tokenizer, passage_id=None, sentence_id=None):
    text=doc['text'][a:b];uid=digest([doc['dataset'],doc['document_id'],kind,a,b,text])[:32]
    return dict(dataset=doc['dataset'],document_id=doc['document_id'],passage_id=passage_id,
        sentence_id=sentence_id,proposition_id=uid if kind=='PROPOSITION' else None,
        unit_id=uid,unit_type=kind,source_text=text,normalized_text=' '.join(text.split()),
        start_char=a,end_char=b,token_count=len(tokenizer.encode(text,add_special_tokens=False)),
        parent_unit_id=parent,document_order=order,hash=digest(text),
        source_support='exact_source_substring',splitter=SPLITTER if kind=='SENTENCE' else METHOD if kind=='PROPOSITION' else None)


def bounded_passages(text, tokenizer, maximum):
    # Stable character spans; exact token counting includes punctuation/spacing.
    # Native passages supersede this chunker. Oversize sentences split at word boundaries.
    import re
    words=list(re.finditer(r'\S+',text));out=[];start=None;end=None
    for match in words:
        if start is None:start=match.start()
        if end is not None and len(tokenizer.encode(text[start:match.end()],add_special_tokens=False))>maximum:
            out.append((start,end));start=match.start()
        end=match.end()
        if len(tokenizer.encode(text[start:end],add_special_tokens=False))>maximum:
            raise ValueError('A single lexical unit exceeds passage token ceiling')
    if start is not None:out.append((start,end))
    return out


def build(documents, tokenizer, config):
    result=[]
    for document_order,doc in enumerate(documents):
        root=unit(doc,'DOCUMENT',0,len(doc['text']),None,[document_order,0,0,0],tokenizer)
        result.append(root)
        passages=doc.get('native_passages') or [dict(start_char=a,end_char=b)
            for a,b in bounded_passages(doc['text'],tokenizer,config['max_passage_tokens'])]
        for pi,p in enumerate(passages):
            a,b=p['start_char'],p['end_char'];passage=unit(doc,'PASSAGE',a,b,root['unit_id'],[document_order,pi,0,0],tokenizer)
            passage['passage_id']=passage['unit_id'];result.append(passage)
            native=[s for s in doc.get('native_sentences',[]) if a<=s['start_char'] and s['end_char']<=b]
            ss=[(s['start_char'],s['end_char']) for s in native] if native else [(a+x,a+y) for x,y in sentence_spans(doc['text'][a:b])]
            for si,(x,y) in enumerate(ss):
                sentence=unit(doc,'SENTENCE',x,y,passage['unit_id'],[document_order,pi,si,0],tokenizer,passage['unit_id'])
                sentence['sentence_id']=sentence['unit_id'];result.append(sentence)
                for ci,(u,v) in enumerate(proposition_spans(doc['text'][x:y])):
                    prop=unit(doc,'PROPOSITION',x+u,x+v,sentence['unit_id'],[document_order,pi,si,ci],
                        tokenizer,passage['unit_id'],sentence['unit_id'])
                    prop.update(source_sentence_id=sentence['unit_id'],source_span=[x+u,x+v],
                        exact_source_support=True,fuzzy_source_support=None,proposition_text=prop['source_text'],
                        proposition_token_count=prop['token_count'])
                    result.append(prop)
    validate(result,documents)
    return result


def validate(units, documents):
    docs={d['document_id']:d for d in documents};known={u['unit_id']:u for u in units}
    if len(known)!=len(units):raise ValueError('Duplicate evidence identities')
    for u in units:
        doc=docs[u['document_id']]
        if doc['text'][u['start_char']:u['end_char']]!=u['source_text'] or digest(u['source_text'])!=u['hash']:
            raise ValueError('Source offset/text/hash mismatch')
        if u['parent_unit_id']:
            p=known[u['parent_unit_id']]
            if p['document_id']!=u['document_id'] or not p['start_char']<=u['start_char']<=u['end_char']<=p['end_char']:
                raise ValueError('Broken parent source linkage')

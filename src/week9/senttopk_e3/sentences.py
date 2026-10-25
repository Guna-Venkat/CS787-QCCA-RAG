"""Academic-rule sentence splitting, exact offsets and query-only ranking v1."""
import re
import numpy as np
from ..provenance import digest,assert_inference_clean
from ..retrieval import unit_vectors

SPLITTER='academic-offset-rules-v1'
ABBREVIATIONS=('e.g.','i.e.','et al.','fig.','figs.','eq.','eqs.','sec.','sect.','ref.','refs.','dr.','prof.','vs.','cf.','approx.','no.','nos.','vol.','pp.','p.','etc.')


def spans(text):
    cuts={0,len(text)}
    # Paragraph breaks are boundaries; single newlines can be wrapped prose.
    cuts.update(m.end() for m in re.finditer(r'\n[ \t]*\n+',text))
    for match in re.finditer(r'[.!?]+[\"\'”’\)\]]*(?=\s|$)',text):
        punctuation=match.group().rstrip('\"\'”’)]')
        prefix=text[:match.start()+len(punctuation)]
        if punctuation=='.' and (any(re.search(r'(?<!\w)'+re.escape(a)+r'$',prefix,re.I) for a in ABBREVIATIONS)
             or re.search(r'\b[A-Z]\.$',prefix) or re.search(r'(?:\b[A-Za-z]\.){2,}$',prefix)):
            continue
        cuts.add(match.end())
    ordered=sorted(cuts);result=[]
    for a,b in zip(ordered,ordered[1:]):
        part=text[a:b];start=a+len(part)-len(part.lstrip());end=b-(len(part)-len(part.rstrip()))
        if end>start and re.search(r'\w',text[start:end]):result.append((start,end))
    return result


def inventory(question_id,paper_id,chunks,tokenizer):
    assert_inference_clean(chunks)
    if len({c['chunk_id'] for c in chunks})!=len(chunks):raise ValueError('Duplicate coarse chunks')
    ordered=sorted(chunks,key=lambda c:(c['char_start'],c['char_end'],c['chunk_id']))
    source_ranks={c['chunk_id']:i for i,c in enumerate(ordered)}
    sentences=[]
    for selected_chunk_rank,c in enumerate(chunks):
        if c['paper_id']!=paper_id:raise ValueError('Cross-paper chunk')
        for order,(a,b) in enumerate(spans(c['text'])):
            text=c['text'][a:b]
            sentences.append(dict(question_id=str(question_id),paper_id=str(paper_id),chunk_id=c['chunk_id'],
                source_chunk_hash=digest(c['text']),sentence_id=digest([paper_id,c['chunk_id'],digest(c['text']),a,b,SPLITTER])[:24],
                char_start=a,char_end=b,document_char_start=c['char_start']+a,document_char_end=c['char_start']+b,
                original_chunk_rank=source_ranks[c['chunk_id']],selected_coarse_chunk_rank=selected_chunk_rank,
                sentence_order=order,text=text,text_hash=digest(text),token_count=len(tokenizer.encode(text,add_special_tokens=False)),
                splitter=SPLITTER,boundary_note='Coarse chunk edges may contain partial original-document sentences.'))
    return sentences


def validate(sentences,chunks,paper_id):
    known={c['chunk_id']:c for c in chunks}
    if len({s['sentence_id'] for s in sentences})!=len(sentences):raise ValueError('Duplicate sentence ID')
    for s in sentences:
        c=known.get(s['chunk_id'])
        if (c is None or s['paper_id']!=paper_id or c['paper_id']!=paper_id or
            s['source_chunk_hash']!=digest(c['text']) or c['text'][s['char_start']:s['char_end']]!=s['text'] or
            digest(s['text'])!=s['text_hash'] or s['document_char_start']!=c['char_start']+s['char_start']):
            raise ValueError('Sentence/source exact identity failure; no replacement allowed')


def rank(question,sentences,query_vector,sentence_vectors):
    # Deliberately no draft, outcomes, evaluation object or oracle input.
    if not isinstance(question,str):raise ValueError('Query must be text only')
    assert_inference_clean(sentences)
    if not sentences:return []
    matrix=unit_vectors(sentence_vectors);q=unit_vectors([query_vector])[0]
    if len(matrix)!=len(sentences):raise ValueError('Sentence/vector alignment mismatch')
    scores=np.clip(matrix@q,-1,1)
    indices=sorted(range(len(sentences)),key=lambda i:(-float(scores[i]),sentences[i]['original_chunk_rank'],sentences[i]['sentence_order'],sentences[i]['sentence_id']))
    return [dict(sentences[i],cosine=float(scores[i]),selected_rank=r+1) for r,i in enumerate(indices)]


def select(ranked,k):
    if k not in (1,2,3,4,6,8):raise ValueError('Only preregistered fixed k values are allowed')
    chosen=ranked[:k]
    presented=sorted(chosen,key=lambda s:(s['document_char_start'],s['original_chunk_rank'],s['sentence_order'],s['sentence_id']))
    return [dict(s,presentation_rank=i+1) for i,s in enumerate(presented)]


def evidence(selected):
    # No labels, facts, paraphrases or blueprint are injected into this block.
    return '\n\n'.join(s['text'] for s in selected)


def diagnostics(selected,ranked,vectors_by_id):
    n=len(selected);scores=[s['cosine'] for s in selected]
    pairs=[(a,b) for i,a in enumerate(selected) for b in selected[i+1:]]
    same=sum(a['chunk_id']==b['chunk_id'] for a,b in pairs)
    adjacent=sum(a['chunk_id']==b['chunk_id'] and abs(a['sentence_order']-b['sentence_order'])==1 for a,b in pairs)
    cosines=[float(np.clip(np.dot(vectors_by_id[a['sentence_id']],vectors_by_id[b['sentence_id']]),-1,1)) for a,b in pairs]
    next_score=ranked[n]['cosine'] if n<len(ranked) else None
    return dict(actual_sentence_k=n,mean_selected_cosine=float(np.mean(scores)) if scores else None,
        min_selected_cosine=min(scores) if scores else None,next_unselected_cosine=next_score,
        selected_vs_next_margin=min(scores)-next_score if scores and next_score is not None else None,
        same_chunk_pair_fraction=same/len(pairs) if pairs else None,
        all_selected_same_chunk=len({s['chunk_id'] for s in selected})==1 if n else None,
        adjacent_sentence_pair_rate=adjacent/len(pairs) if pairs else None,
        mean_pairwise_selected_cosine=float(np.mean(cosines)) if pairs else None,
        exact_duplicate_sentence_pair_rate=sum(a['text']==b['text'] for a,b in pairs)/len(pairs) if pairs else None,
        near_duplicate_sentence_pair_rate=sum(v>=.95 for v in cosines)/len(pairs) if pairs else None)

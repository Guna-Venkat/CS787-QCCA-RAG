"""Diagnostic source-offset retention; annotation unions are not necessity labels."""
def merge(intervals):
    out=[]
    for a,b in sorted(set(tuple(x) for x in intervals if x[1]>x[0])):
        if out and a<=out[-1][1]:out[-1][1]=max(out[-1][1],b)
        else:out.append([a,b])
    return out

def overlap(a,b,intervals):
    return sum(max(0,min(b,y)-max(a,x)) for x,y in merge(intervals))

def annotations(example):
    docs={d['document_id']:d['text'] for d in example['inference']['documents']}
    result=[];seen=set()
    for g in example['evaluation'].get('gold_spans',[]):
        key=(g.get('document_id'),g.get('evidence_id'),g.get('text'))
        if key in seen:continue
        seen.add(key);did=g.get('document_id');candidates=[]
        if g.get('start_char') is not None and g.get('end_char') is not None:
            candidates.append((g['start_char'],g['end_char']))
        candidates+=list(map(tuple,g.get('alternatives',[])))
        valid=sorted({(a,b) for a,b in candidates if did in docs and 0<=a<b<=len(docs[did])})
        result.append(dict(evidence_id=g.get('evidence_id'),document_id=did,text=g.get('text',''),
            alternatives=valid,mapped=bool(valid)))
    return result

def retention(example,units,tok):
    """Full-span recall, annotated-document recall, fully covered gold token fraction.

    Each QASPER annotation uses its best matching alternative occurrence. Different
    annotations are pooled (not a minimal jointly necessary set). Unmapped spans
    and absent NEI evidence are explicit, never automatically successes/misses.
    """
    gs=annotations(example);mapped=[g for g in gs if g['mapped']]
    docs={d['document_id']:d['text'] for d in example['inference']['documents']}
    selected={did:merge([(u['start_char'],u['end_char']) for u in units if u['document_id']==did])
              for did in {u['document_id'] for u in units}}
    full=0;char_covered=0;char_total=0;token_covered=0;token_total=0;details=[]
    fast=bool(getattr(tok,'is_fast',False))
    for g in mapped:
        alternatives=[]
        for a,b in g['alternatives']:
            source=docs[g['document_id']][a:b];intervals=selected.get(g['document_id'],[])
            # Omitted inter-sentence whitespace is not lost factual evidence.
            positions={i for i,c in enumerate(source) if not c.isspace()}
            covered={i for i in positions if any(x<=a+i<y for x,y in intervals)}
            hits=len(covered);ct=0;nt=0
            if fast:
                offsets=tok(source,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
                for x,y in offsets:
                    content=positions.intersection(range(x,y))
                    if not content:continue
                    nt+=1;ct+=content<=covered
            alternatives.append(dict(start=a,end=b,chars=hits,total_chars=len(positions),tokens=ct,total_tokens=nt))
        best=min(alternatives,key=lambda x:(-x['chars']/max(1,x['total_chars']),-x['tokens'],x['start']))
        is_full=bool(best['total_chars']) and best['chars']==best['total_chars'];full+=is_full
        char_covered+=best['chars'];char_total+=best['total_chars']
        token_covered+=best['tokens'];token_total+=best['total_tokens']
        details.append(dict(evidence_id=g['evidence_id'],document_id=g['document_id'],fully_retained=is_full,
            char_fraction=best['chars']/best['total_chars'] if best['total_chars'] else None,token_fraction=best['tokens']/best['total_tokens'] if best['total_tokens'] else None))
    parents={g['document_id'] for g in mapped};represented={u['document_id'] for u in units}
    return dict(annotation_count=len(gs),mapped_annotations=len(mapped),unmapped_annotations=len(gs)-len(mapped),
        gold_evidence_recall=full/len(mapped) if mapped else None,
        gold_parent_recall=len(parents&represented)/len(parents) if parents else None,
        gold_evidence_char_coverage=char_covered/char_total if char_total else None,
        gold_evidence_token_coverage=token_covered/token_total if token_total else None,
        token_coverage_available=fast and bool(token_total),annotation_details=details,
        gold_parent_definition='annotated native document, not inferred necessary coarse chunk',
        whitespace_policy='source offsets retained exactly; only non-whitespace content determines span/token coverage',
        annotation_scope='deduplicated pooled support annotations; alternative occurrences use best source match')

def retrieval_status(ret):
    if not ret['mapped_annotations']:return 'ANNOTATIONS_UNAVAILABLE'
    if ret['gold_evidence_recall']==1 and ret['unmapped_annotations']==0:return 'RETRIEVAL_SUFFICIENT'
    if ret['gold_evidence_char_coverage']>0:return 'RETRIEVAL_PARTIAL'
    return 'RETRIEVAL_MISS' if not ret['unmapped_annotations'] else 'ANNOTATIONS_INCOMPLETE'

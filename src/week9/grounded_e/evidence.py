"""Exact-offset grounding and whole-representation token budgets; no fuzzy repair."""
import json
import re
from ..provenance import digest, assert_inference_clean
from ..retrieval import bm25


def tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def locate_support(source, proposed):
    # Strict substring matching is deliberately stronger than optional formatting
    # normalization. Never convert a fuzzy/semantic match into alleged source text.
    if not isinstance(proposed, str) or not proposed.strip():
        return None
    start = source.find(proposed)
    return None if start < 0 else (start, start + len(proposed), source[start:start + len(proposed)])


def render(policy, items):
    if policy == 'E0':
        return '\n'.join('[' + i['source_chunk_id'] + '] ' + i['support_span'] for i in items)
    key = 'normalized_claim' if policy == 'E2' else 'fact'
    fields = ['source_chunk_id', 'support_span', key] + (['evidence_type', 'relevance'] if policy == 'E2' else [])
    return json.dumps([{k:i[k] for k in fields} for i in items], ensure_ascii=False, separators=(',', ':')) if items else ''


def pack(policy, proposals, chunks, qid, tokenizer, budget, *, invalid_format=False, empty=False, cap_hit=False):
    if budget not in (64, 128):
        raise ValueError('Preregistered evidence budgets: 64 or 128 tokens')
    known = {c['chunk_id']:c for c in chunks}
    if len(known) != len(chunks):
        raise ValueError('Duplicate selected chunks')
    accepted=[];rejected=[];supported=0;seen=set()
    for obj in proposals:
        if not isinstance(obj, dict):
            rejected.append(dict(reason='item_not_object'));continue
        chunk = known.get(obj.get('source_chunk_id'))
        match = locate_support(chunk['text'], obj.get('support_span')) if chunk else None
        if match is None:
            rejected.append(dict(reason='unsupported_span_or_source', proposed=obj));continue
        supported += 1
        start,end,span=match
        item=dict(source_chunk_id=chunk['chunk_id'],source_paper_id=chunk['paper_id'],question_id=str(qid),char_start=start,char_end=end,support_span=span,exact_source_text=span,support_valid=True,support_tokens=tokens(tokenizer,span),token_count=tokens(tokenizer,span))
        if policy != 'E0':
            key = 'normalized_claim' if policy == 'E2' else 'fact'
            fact=obj.get(key)
            # A quote alone does NOT validate a paraphrased fact. Use the
            # conservative copied-subspan variant rather than claim entailment.
            fact_match=locate_support(span,fact)
            if fact_match is None or tokens(tokenizer,fact)>12 or tokens(tokenizer,span)>24:
                rejected.append(dict(reason='fact_not_copied_from_support_or_item_too_long',proposed=obj));continue
            item[key]=' '.join(fact_match[2].split())
            item['fact_tokens']=tokens(tokenizer,item[key])
            item['fact_support_mode']='copied_contiguous_subspan_whitespace_normalized'
            if policy == 'E2':
                if obj.get('evidence_type') not in ('answer_detail','correction','qualification','definition') or obj.get('relevance') not in ('direct','missing','conflicting'):
                    rejected.append(dict(reason='invalid_blueprint_labels',proposed=obj));continue
                item.update(evidence_type=obj['evidence_type'],relevance=obj['relevance'])
        key=(item['source_chunk_id'],start,end)
        if key in seen:
            rejected.append(dict(reason='duplicate_span',proposed=obj));continue
        if len(accepted)>=2 or tokens(tokenizer,render(policy,accepted+[item]))>budget:
            rejected.append(dict(reason='evidence_budget_or_item_limit',proposed=obj));continue
        accepted.append(item);seen.add(key)
    evidence=render(policy,accepted)
    count=tokens(tokenizer,evidence)
    assert count<=budget
    failed=not accepted or invalid_format
    if failed:
        accepted=[];evidence='';count=0
    reason=('malformed_output' if invalid_format else 'empty_extraction' if empty else 'all_items_rejected_or_over_budget') if failed else None
    return dict(text=evidence,items=accepted,rejected_items=rejected,evidence_tokens=count,
        support_tokens=sum(i['support_tokens'] for i in accepted),fact_tokens=sum(i.get('fact_tokens',0) for i in accepted),
        num_supported_items=supported,num_accepted_items=len(accepted),num_rejected_items=len(rejected),
        num_accepted_facts=len(accepted) if policy!='E0' else 0,num_rejected_facts=len(rejected) if policy!='E0' else 0,
        exact_span_support_rate=supported/len(proposals) if proposals else None,
        accepted_exact_span_support_rate=1.0 if accepted else None,
        zero_exact_support=supported==0,invalid_format=invalid_format,empty_extraction=empty,
        fail_closed=failed,failure_reason=reason,extraction_valid=not failed,reached_extraction_token_limit=cap_hit,
        evidence_hash=digest(evidence),max_evidence_tokens=budget,
        grounding_note='Exact source support is mechanical, not semantic entailment. Facts are conservatively copied subspans, not free paraphrases.')


def parse(policy, raw, chunks, qid, tokenizer, budget, cap_hit=False):
    if policy not in ('E1-Q','E1-QD','E2'):
        raise ValueError('Unknown structured policy')
    try:
        objects=json.loads(raw)
    except (json.JSONDecodeError,TypeError):
        return pack(policy,[],chunks,qid,tokenizer,budget,invalid_format=bool(str(raw).strip()),empty=not str(raw).strip(),cap_hit=cap_hit)
    fields={'source_chunk_id','support_span',('normalized_claim' if policy=='E2' else 'fact')}
    if policy=='E2':fields|={'evidence_type','relevance'}
    valid=isinstance(objects,list) and all(isinstance(o,dict) and set(o)==fields and all(isinstance(v,str) for v in o.values()) for o in objects)
    if not valid:
        return pack(policy,[],chunks,qid,tokenizer,budget,invalid_format=True,cap_hit=cap_hit)
    return pack(policy,objects,chunks,qid,tokenizer,budget,empty=not objects,cap_hit=cap_hit)


def extractive(inference,chunks,tokenizer,budget):
    assert_inference_clean(inference);assert_inference_clean(chunks)
    spans=[]
    for chunk in chunks:
        for match in re.finditer(r'[^\n.!?]+(?:[.!?](?=\s|$)|(?=\n)|$)',chunk['text']):
            raw=match.group();left=len(raw)-len(raw.lstrip());text=raw.strip()
            if text and re.search(r'\w',text):
                spans.append((chunk,match.start()+left,text))
    scores=bm25(inference['question'],[s[2] for s in spans]) if spans else []
    order=sorted(range(len(spans)),key=lambda i:(-scores[i],spans[i][0]['chunk_id'],spans[i][1]))
    proposals=[]
    for idx in order:
        chunk,start,text=spans[idx]
        # Trim only at exact source word boundaries. Include all representation
        # overhead in the budget and never decode/rewrite source characters.
        ends=[m.end() for m in re.finditer(r'\S+',text)]
        lo,hi=0,len(ends);best=None
        while lo<hi:
            mid=(lo+hi)//2
            obj=dict(source_chunk_id=chunk['chunk_id'],support_span=text[:ends[mid]])
            candidate=proposals+[obj]
            if tokens(tokenizer,render('E0',candidate))<=budget:
                best=obj;lo=mid+1
            else:hi=mid
        if best:
            proposals.append(best)
        if len(proposals)==2:break
    return pack('E0',proposals,chunks,inference['question_id'],tokenizer,budget,empty=not proposals)

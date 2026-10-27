"""A shared textual selection must fit every pinned reader's tokenizer budget."""
import re


class FixtureTokenizer:
    name_or_path='SYNTHETIC_WORD_TOKENIZER'
    def encode(self,text,add_special_tokens=False):return re.findall(r'\w+|[^\w\s]',text)


def source_order(units):
    return sorted(units,key=lambda u:(u['document_order'],u['unit_id']))


def evidence(units):return '\n\n'.join(u['source_text'] for u in source_order(units))


def token_counts(text,tokenizers):
    return {k:len(t.encode(text,add_special_tokens=False)) for k,t in tokenizers.items()}


def allocate(ranked,budget,tokenizers):
    selected=[];seen=set()
    for u in ranked:
        if u['unit_id'] in seen:continue
        # Hierarchical strategies must not double-expose overlapping parent/child spans.
        if any(u['document_id']==v['document_id'] and u['start_char']<v['end_char'] and v['start_char']<u['end_char'] for v in selected):continue
        trial=selected+[u]
        if budget!='FULL' and max(token_counts(evidence(trial),tokenizers).values(),default=0)>budget:
            break  # preregistered strict-prefix rule, never silently skip oversize units
        selected=trial;seen.add(u['unit_id'])
    text=evidence(selected);counts=token_counts(text,tokenizers)
    return dict(text=text,units=source_order(selected),selected_unit_ids=[u['unit_id'] for u in source_order(selected)],
        selection_rank_ids=[u['unit_id'] for u in selected],token_counts=counts,requested_budget=budget,
        actual_evidence_tokens=counts.get('Qwen',next(iter(counts.values()),0)),
        unused_budget=None if budget=='FULL' else budget-max(counts.values(),default=0),
        budget_policy='strict rank prefix; maximum actual count across all three reader tokenizers',source_support=1.)

"""Inference-only features of the specific (question, y0, evidence, F0 answer) tuple."""
import re
from collections import Counter
import numpy as np
import pandas as pd
from ..provenance import assert_inference_clean

GROUPS={
    'V0':['normalized_equal','token_edit_distance','token_jaccard','added_token_count','removed_token_count',
          'initial_answer_length','candidate_answer_length','answer_length_difference','answer_length_ratio',
          'changed_fraction','one_answer_substring'],
    'V1_lexical':['initial_evidence_overlap','candidate_evidence_overlap','evidence_overlap_delta',
                  'added_tokens_in_evidence_fraction','removed_tokens_absent_evidence_fraction'],
    'V2_length':['initial_generated_tokens','candidate_generated_tokens'],
    'V3':['top_sentence_cosine','second_sentence_cosine','third_sentence_cosine','mean_selected_cosine',
          'min_selected_cosine','next_unselected_cosine','selected_vs_next_margin','evidence_tokens',
          'mean_source_chunk_rank','selected_chunk_count'],
    'V4':['query_tokens','question_mark','starts_wh_question','starts_auxiliary_question'],
}
PAYLOAD_KEYS={'question','initial_answer','candidate_answer','evidence_text','ranking','generation_lengths'}
RANK_KEYS={'sentence_cosines','mean_selected_cosine','min_selected_cosine','next_unselected_cosine',
           'selected_vs_next_margin','evidence_tokens','source_chunk_ranks','selected_chunk_count'}
LENGTH_KEYS={'initial_generated_tokens','candidate_generated_tokens'}


def tokens(text):return re.findall(r'\w+',text.lower())


def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def payload(d0,f0,src):
    # Explicit projection: never forward whole outcome records to an inference API.
    return dict(question=f0['question'],initial_answer=d0['prediction'],candidate_answer=f0['prediction'],
        evidence_text=f0['evidence_text'],
        ranking=dict(sentence_cosines=[s['cosine'] for s in sorted(src['selected_sentences'],key=lambda s:s['selected_rank'])],
            mean_selected_cosine=src['mean_selected_cosine'],min_selected_cosine=src['min_selected_cosine'],
            next_unselected_cosine=src['next_unselected_cosine'],selected_vs_next_margin=src['selected_vs_next_margin'],
            evidence_tokens=src['evidence_tokens'],source_chunk_ranks=[s['original_chunk_rank'] for s in src['selected_sentences']],
            selected_chunk_count=len(set(s['chunk_id'] for s in src['selected_sentences']))),
        generation_lengths=dict(initial_generated_tokens=f0.get('initial_draft_generated_tokens'),
            candidate_generated_tokens=f0.get('revision_generated_tokens')))


def online_features(item):
    assert_inference_clean(item)
    if set(item)!=PAYLOAD_KEYS or set(item['ranking'])!=RANK_KEYS or set(item['generation_lengths'])!=LENGTH_KEYS:
        raise ValueError('Strict action-feature schema: outcome/ID/unknown fields are forbidden')
    a=tokens(item['initial_answer']);b=tokens(item['candidate_answer']);e=set(tokens(item['evidence_text']))
    sa,sb=set(a),set(b);ca,cb=Counter(a),Counter(b)
    added=list((cb-ca).elements());removed=list((ca-cb).elements())
    overlap_a=len(sa&e)/len(sa) if sa else 0.
    overlap_b=len(sb&e)/len(sb) if sb else 0.
    distance=edit_distance(a,b);r=item['ranking'];cos=r['sentence_cosines'];q=tokens(item['question'])
    def available(x):return float(x) if x is not None else np.nan
    result=dict(normalized_equal=float(a==b),token_edit_distance=distance,
        token_jaccard=len(sa&sb)/len(sa|sb) if sa|sb else 1.,added_token_count=len(added),removed_token_count=len(removed),
        initial_answer_length=len(a),candidate_answer_length=len(b),answer_length_difference=len(b)-len(a),
        answer_length_ratio=len(b)/max(len(a),1),changed_fraction=distance/max(len(a),len(b),1),
        one_answer_substring=float(bool(a and b) and (' '.join(a) in ' '.join(b) or ' '.join(b) in ' '.join(a))),
        initial_evidence_overlap=overlap_a,candidate_evidence_overlap=overlap_b,evidence_overlap_delta=overlap_b-overlap_a,
        added_tokens_in_evidence_fraction=sum(t in e for t in added)/len(added) if added else 0.,
        removed_tokens_absent_evidence_fraction=sum(t not in e for t in removed)/len(removed) if removed else 0.,
        top_sentence_cosine=available(cos[0]) if len(cos)>0 else np.nan,
        second_sentence_cosine=available(cos[1]) if len(cos)>1 else np.nan,
        third_sentence_cosine=available(cos[2]) if len(cos)>2 else np.nan,
        **{k:available(r[k]) for k in ('mean_selected_cosine','min_selected_cosine','next_unselected_cosine','selected_vs_next_margin','evidence_tokens')},
        mean_source_chunk_rank=float(np.mean(r['source_chunk_ranks'])) if r['source_chunk_ranks'] else np.nan,
        selected_chunk_count=r['selected_chunk_count'],
        query_tokens=len(q),question_mark=float(item['question'].rstrip().endswith('?')),
        starts_wh_question=float(bool(q) and q[0] in ('what','which','who','whom','whose','where','when','why','how')),
        starts_auxiliary_question=float(bool(q) and q[0] in ('is','are','was','were','do','does','did','can','could','has','have','had','will','would','should')),
        **{k:available(item['generation_lengths'][k]) for k in LENGTH_KEYS})
    if set(result)!=set(sum(GROUPS.values(),[])):raise ValueError('Feature registry mismatch')
    return result


def feature_matrix(records,groups):
    allowed=set(sum(GROUPS.values(),[]))
    if any(set(r)!=allowed for r in records):raise ValueError('Feature matrix contains missing/extra columns; outcomes/IDs cannot be predictors')
    if not groups or any(g not in GROUPS for g in groups):raise ValueError('Unknown/empty feature ablation')
    columns=[name for g in groups for name in GROUPS[g]]
    assert_inference_clean(dict.fromkeys(columns))
    frame=pd.DataFrame(records)[columns].astype(float)
    if np.isinf(frame.to_numpy()).any():raise ValueError('Infinite online feature')
    return frame


def prepare(bundle,cfg):
    records=[];meta=[]
    for q in sorted(bundle['groups']['D0']):
        d0=bundle['groups']['D0'][q];f0=bundle['groups']['F0'][q]
        records.append(online_features(payload(d0,f0,bundle['e3'][q])))
        meta.append(dict(question_id=q,paper_id=d0['paper_id']))
    return feature_matrix(records,cfg['feature_groups']),pd.DataFrame(meta)


AVAILABILITY=dict(V0='Available from actual y0/y1 answer strings',V1_lexical='Available from frozen E3 evidence',
    V1_semantic='Not computed: q/answer/evidence MiniLM similarities are optional future features; no encoder loaded by preflight/scaffold.',
    V2_confidence='Unavailable: saved outputs lack reliable logprob/entropy/EOS fields. No generation backfill.',
    V2_length='Only saved generation lengths, not confidence measurements',
    V3='Existing E3 rank/cosine/length data; raw chunk/paper IDs are metadata, never numeric predictors',
    V4='Generic deterministic question form and length; no gold answer types')

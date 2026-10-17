"""Checks of draft-only targets, real slot interventions, and W8-D cache validity."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from src.week8.attribution import (
    build_draft_teacher_forcing_inputs, compute_masking_attribution,
    compute_masking_attribution_fallback, extract_generator_interaction_features,
)
from src.week8.generator_features import validate_feature_rows


class Tokenizer:
    def __call__(self, text, return_tensors='pt', add_special_tokens=True):
        # The boundary would change if prompt and answer were tokenized together.
        ids = [1, 10, 11] if text == 'prompt' else [20, 21]
        assert text in ('prompt', ' draft')
        if text == ' draft':
            assert not add_special_tokens
        return {'input_ids': torch.tensor([ids])}


class Model(torch.nn.Module):
    dtype = torch.float32
    def __init__(self):
        super().__init__()
        self.calls = []
    def forward(self, input_ids, retrieval_embeds, labels, **kwargs):
        self.calls.append((input_ids.clone(), retrieval_embeds.clone(), labels.clone()))
        assert labels.tolist() == [[-100, -100, -100, 20, 21]]
        # Baseline NLL = 10 - 1 - 3 = 6; masking each slot raises NLL
        # by its distinct contribution. Non-target prompt IDs never contribute.
        return SimpleNamespace(loss=10 - retrieval_embeds[:, 0].sum())


class Encoder:
    def encode(self, texts, **kwargs):
        return np.tile([1., 0.], (len(texts), 1))


def test_teacher_forcing_keeps_prompt_boundary_and_masks_only_prompt():
    ids, labels, count = build_draft_teacher_forcing_inputs(Tokenizer(), 'prompt', 'draft')
    assert ids.tolist() == [[1, 10, 11, 20, 21]]
    assert labels.tolist() == [[-100, -100, -100, 20, 21]]
    assert count == 2
    with pytest.raises(ValueError, match='empty draft'):
        build_draft_teacher_forcing_inputs(Tokenizer(), 'prompt', '')


def test_masking_uses_requested_slot_positions_and_correct_sum_sign():
    model = Model().eval()
    embeds = torch.tensor([[1., 7.], [3., 8.]])
    original = embeds.clone()
    result = compute_masking_attribution(model, Tokenizer(), 'prompt', 'draft', embeds, [1, 0], 'cpu')
    assert result == {1: 6., 0: 2.}
    assert torch.equal(embeds, original)
    assert model.calls[1][1].tolist() == [[1., 7.], [0., 0.]]
    assert model.calls[2][1].tolist() == [[0., 0.], [3., 8.]]


def test_masking_rejects_unsupervised_or_invalid_slot_requests():
    model = Model().eval()
    ids, labels, _ = build_draft_teacher_forcing_inputs(Tokenizer(), 'prompt', 'draft')
    embeds = torch.ones(2, 2)
    with pytest.raises(ValueError, match='labels'):
        compute_masking_attribution_fallback(model, ids, embeds, [0], 'cpu')
    with pytest.raises(ValueError, match='indices'):
        compute_masking_attribution_fallback(model, ids, embeds, [2], 'cpu', labels=labels)
    with pytest.raises(ValueError, match='No draft target'):
        compute_masking_attribution_fallback(model, ids, embeds, [0], 'cpu', labels=torch.full_like(ids, -100))


def test_no_fake_attention_and_empty_draft_masking_is_unavailable():
    model = Model().eval()
    scorer = SimpleNamespace(encoder=Encoder())
    features = extract_generator_interaction_features(model, Tokenizer(), 'prompt', '',
        torch.ones(2, 2), {5:'doc', 6:'other'}, {5:1, 6:0}, scorer, 'cpu')
    assert not model.calls
    for row in features.values():
        assert not any(k.startswith(('a1_', 'a2_', 'a3_', 'a4_', 'a5_')) for k in row)
        assert row['masking_available'] == 0
        assert row['masking_target_tokens'] == 0
        assert np.isnan(row['masking_delta_logp'])
        assert row['a6_draft_sim'] == pytest.approx(1.)


def test_cache_rejects_changed_routing_or_sources():
    import hashlib
    group = pd.DataFrame([dict(question_id='q', paper_id='p', draft_answer='',
                              semantic_rank=r, candidate_passage_index=r-1) for r in range(5,11)])
    rows = [dict(question_id='q',paper_id='p',semantic_rank=r,candidate_passage_index=r-1,
             source_fingerprint='v1',draft_sha256=hashlib.sha256(b'').hexdigest(),
             masking_available=0,masking_target_tokens=0,masking_delta_logp=None,
             masking_delta_logp_per_token=None,query_relevance_rq=.2,draft_relevance_ry=.2,
             redundancy_red=.2,candidate_diversity=.2,mig_score=.2,a6_draft_sim=.2) for r in range(5,11)]
    validate_feature_rows(rows, group, 'v1')
    with pytest.raises(ValueError, match='sources/candidate'):
        validate_feature_rows(rows,group,'v2')
    with pytest.raises(ValueError, match='six'):
        validate_feature_rows(rows[:-1],group,'v1')
    rows[0]['candidate_passage_index']=0
    with pytest.raises(ValueError, match='sources/candidate'):
        validate_feature_rows(rows,group,'v1')


def test_query_features_keep_canonical_draft_and_map_original_indices_to_slots(monkeypatch):
    import src.week8.generator_features as gf
    candidates = [8, 4, 6, 1, 0, 9]
    full = [2, 3, 5, 7]
    scores = np.ones(10)
    scores[candidates] = [.6, .5, .4, .3, .2, .1]
    group = pd.DataFrame([dict(question_id='q', paper_id='p', draft_answer='actual draft',
        semantic_rank=r, candidate_passage_index=i, candidate_text=f'doc {i}', semantic_score=scores[i])
        for r,i in zip(range(5,11),candidates)])
    record = dict(question_id='q',question='question',gold_answers=['DO NOT USE GOLD'],
                  passages=[{'text':f'doc {i}'} for i in range(10)])
    def prompt(question, passages, full_indices, medium, heavy, extracts, tokenizer, **kwargs):
        assert full_indices == full
        assert heavy == sorted(candidates)
        assert medium == []
        return 'prompt', 100, heavy, {}
    def mig(question, draft, full_texts, candidate_texts, scorer):
        assert draft == 'actual draft'
        assert full_texts == [f'doc {i}' for i in full]
        return {r:dict(query_relevance_rq=.2,draft_relevance_ry=.2,
             redundancy_red=.2,candidate_diversity=.2,mig_score=.2) for r in candidate_texts}
    def interaction(model, tokenizer, prompt, draft, embeds, texts, slots, scorer, device):
        assert draft == 'actual draft'
        assert slots == {r:sorted(candidates).index(i) for r,i in zip(range(5,11),candidates)}
        assert torch.equal(embeds[:,0], torch.tensor(sorted(candidates),dtype=torch.float32))
        return {r:dict(masking_delta_logp=1.,masking_delta_logp_per_token=.5,
                       masking_target_tokens=2,masking_available=1,a6_draft_sim=.2) for r in texts}
    monkeypatch.setattr(gf,'build_mixed_fidelity_sara_prompt',prompt)
    monkeypatch.setattr(gf,'compute_mig_features_for_query',mig)
    monkeypatch.setattr(gf,'extract_generator_interaction_features',interaction)
    scorer = SimpleNamespace(score_query_passages=lambda *args: torch.tensor(scores))
    rows = gf.compute_query_features(group,record,torch.arange(10,dtype=torch.float32).reshape(10,1),
                                    None,None,scorer,'v1','cpu')
    assert [r['candidate_passage_index'] for r in rows] == candidates
    assert all('gold_answers' not in row and 'delta_f1' not in row for row in rows)


def test_canonical_context_strips_question_like_original_generator(monkeypatch):
    import src.week8.generator_features as gf
    group=pd.DataFrame([dict(question_id='q',question='question?',paper_id='p',
        semantic_rank=r,candidate_passage_index=r-1,base_context_tokens=100) for r in range(5,11)])
    record={'question':'  question? \n','passages':[{'text':f'doc {i}'} for i in range(10)]}
    def prompt(question,passages,full,medium,heavy,extracts,tokenizer,**kwargs):
        assert question=='question?'
        assert full==[0,1,2,3] and heavy==[4,5,6,7,8,9]
        return 'canonical prompt',100,heavy,{}
    monkeypatch.setattr(gf,'build_mixed_fidelity_sara_prompt',prompt)
    q,p,t,indices,embeds,full=gf.canonical_k4_context(group,record,torch.ones(10,2),None)
    assert q=='question?' and p=='canonical prompt' and t==100
    assert indices==[4,5,6,7,8,9] and embeds.shape==(6,2)
    with pytest.raises(ValueError,match='token count'):
        gf.canonical_k4_context(group.assign(base_context_tokens=101),record,torch.ones(10,2),None)

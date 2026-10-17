"""Regression checks for the W8-C execution and research-validity blockers."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.week8.utility_dataset import build_utility_dataset, get_non_leaking_feature_columns, assert_no_feature_leakage
from src.week8.utility_models import S0_Rank5Selector, S2_SemanticSelector, OracleSelector, truncate_and_budget_tokens
from src.week8.evaluation import compute_query_grouped_selector_metrics
from src.week8.utility_training import validate_surface, assert_training_split, listwise_text


@pytest.fixture
def surface():
    return pd.DataFrame([
        dict(question_id='q', paper_id='p', semantic_rank=r, candidate_passage_index=i,
             question='question', candidate_text=f'evidence {r}', base_prediction='draft',
             base_f1=.2, promoted_f1=.2+d, delta_f1=d, semantic_score=1/r,
             promotion_latency=100*d, answer_changed=int(d != 0), delta_ll=d)
        for r,i,d in zip(range(5,11), [9,4,0,1,8,7], [.1,.1,0,-.1,0,0])])


def test_saved_routing_preserves_original_full_set_despite_ties(surface):
    from src.week8.utility_training import _validate_saved_routing
    passages = ['duplicate'] * 10
    group = surface.assign(candidate_text='duplicate', semantic_score=.2)
    scores = np.full(10, .2)
    scores[7] += 2e-9
    assert _validate_saved_routing(group, passages, scores) == [2, 3, 5, 6]
    # Query 1599: one real passage, nine duplicates; CPU/GPU change tie order.
    group = group.assign(candidate_passage_index=list(range(4, 10)), semantic_score=-.021127505)
    scores = np.array([.199381247] + [-.021127593]*7 + [-.021127591]*2)
    assert _validate_saved_routing(group, passages, scores) == [0, 1, 2, 3]


def test_saved_routing_rejects_text_score_and_rank_drift(surface):
    from src.week8.utility_training import _validate_saved_routing
    passages = ['duplicate'] * 10
    group = surface.assign(candidate_text='duplicate', semantic_score=.2)
    scores = np.full(10, .2)
    bad = group.copy()
    bad.loc[bad.index[0], 'candidate_text'] = 'different passage'
    with pytest.raises(ValueError, match='candidate text'):
        _validate_saved_routing(bad, passages, scores)
    with pytest.raises(ValueError, match='saved MiniLM scores'):
        _validate_saved_routing(group.assign(semantic_score=.3), passages, scores)
    bad_scores = scores.copy()
    bad_scores[7] = .4
    bad = group.copy()
    bad.loc[bad.candidate_passage_index == 7, 'semantic_score'] = .4
    with pytest.raises(ValueError, match='candidate ranks'):
        _validate_saved_routing(bad, passages, bad_scores)
    bad_scores = scores.copy()
    bad_scores[2] = .1
    with pytest.raises(ValueError, match='candidate ranks'):
        _validate_saved_routing(group, passages, bad_scores)


def test_rank_is_not_original_passage_index(surface):
    assert S0_Rank5Selector().predict_scores(surface).argmax() == 0
    assert S2_SemanticSelector().predict_scores(surface).argmax() == 0
    shuffled = surface.iloc[::-1]
    assert shuffled.iloc[S0_Rank5Selector().predict_scores(shuffled).argmax()].semantic_rank == 5


def test_feature_allowlist_excludes_post_promotion_and_score_columns(surface):
    df = build_utility_dataset(surface).assign(score_u0=1., promoted_context_tokens=1200)
    cols = get_non_leaking_feature_columns(df)
    assert not {'promotion_latency','answer_changed','delta_ll','score_u0','promoted_context_tokens'} & set(cols)
    for col in ['base_f1','delta_ll','target_regret','promotion_latency','score_u0']:
        with pytest.raises(AssertionError): assert_no_feature_leakage([col])


def test_metrics_use_current_split_and_credit_tied_best(surface):
    df = surface.assign(score=[0,1,0,0,0,0])
    m = compute_query_grouped_selector_metrics(df, selector_scores_col='score')
    assert m['top1_oracle_accuracy'] == 1
    assert m['base_k4_f1'] == pytest.approx(.2)
    assert m['final_downstream_f1'] == pytest.approx(.3)
    assert m['selector_headroom_recovery'] == pytest.approx(1)
    oracle = compute_query_grouped_selector_metrics(df, selector_obj=OracleSelector())
    assert oracle['mean_regret'] == 0


def test_no_headroom_and_bad_scores(surface):
    df = surface.assign(delta_f1=0, promoted_f1=.2, score=0.)
    assert compute_query_grouped_selector_metrics(df, selector_scores_col='score')['selector_headroom_recovery'] == 0
    df.loc[0,'score'] = np.nan
    with pytest.raises(ValueError): compute_query_grouped_selector_metrics(df, selector_scores_col='score')


def test_surface_rejects_partial_duplicate_and_wrong_paper(surface):
    manifest = pd.DataFrame([{'question_id':'q','paper_id':'p'}])
    validate_surface(surface, manifest)
    for bad in [surface.iloc[:-1], pd.concat([surface,surface.iloc[[0]]]), surface.assign(paper_id='wrong')]:
        with pytest.raises(ValueError): validate_surface(bad, manifest)


def test_drafts_must_cover_queries_and_match_canonical(surface):
    with pytest.raises(ValueError):
        build_utility_dataset(surface, pd.DataFrame([{'question_id':'q','draft_answer':'other'}]))
    with pytest.raises(ValueError):
        build_utility_dataset(surface, pd.DataFrame([{'question_id':'missing','draft_answer':'draft'}]))


def test_no_training_on_internal_val():
    from src.week8.data import load_config, PROJECT_ROOT
    cfg = load_config()
    val = pd.read_csv(PROJECT_ROOT / cfg['splits']['internal_val_manifest'], dtype=str).head(1)
    with pytest.raises(ValueError, match='FIT manifest'): assert_training_split(val)


class WordTokenizer:
    def __init__(self): self.words = {}; self.reverse = {}
    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in str(text).split():
            if word not in self.words:
                idx = len(self.words)+1; self.words[word] = idx; self.reverse[idx] = word
            ids.append(self.words[word])
        return ids
    def decode(self, ids, **kwargs): return ' '.join(self.reverse[i] for i in ids)


def test_budget_caps_draft_even_when_sequence_fits():
    tok = WordTokenizer()
    text, diag = truncate_and_budget_tokens(tok,'query','draft '*150,'evidence '*30)
    assert diag['draft_tokens'] <= 96
    assert diag['candidate_tokens'] == 30
    assert diag['total_tokens'] <= 512


def test_listwise_retains_all_six_candidates(surface):
    tok = WordTokenizer()
    group = build_utility_dataset(surface).assign(candidate_text=['word '*1000]*6)
    text = listwise_text(tok, group)
    assert len(tok.encode(text)) <= 512
    assert all(f'[CANDIDATE {r}]' in text for r in range(5,11))


def test_oof_keeps_all_paper_candidates_together_and_restores_row_order(surface, monkeypatch):
    import src.week8.utility_training as training
    from src.week8.data import PROJECT_ROOT, load_config
    cfg = load_config()
    manifest = pd.read_csv(PROJECT_ROOT / cfg['splits']['fit_manifest'], dtype=str).drop_duplicates('paper_id').head(3)
    blocks = []
    for row in manifest.itertuples():
        blocks.append(surface.assign(question_id=row.question_id, paper_id=row.paper_id))
    df = build_utility_dataset(pd.concat(blocks, ignore_index=True)).sample(frac=1, random_state=12).reset_index(drop=True)
    seen = []
    def fit(name, train, features, settings):
        assert all(len(g) == 6 for _,g in train.groupby('question_id'))
        return set(train.paper_id)
    def predict(train_papers, val, settings):
        assert not train_papers & set(val.paper_id)
        assert all(len(g) == 6 for _,g in val.groupby('question_id'))
        seen.extend(val.question_id.tolist())
        return {'utility_score': val.semantic_rank.to_numpy(float) / 10}
    monkeypatch.setattr(training, 'fit_neural', fit)
    monkeypatch.setattr(training, 'predict_neural', predict)
    predictions, metrics = training.run_oof('U2-F', df, ['semantic_rank'], training.TrainingConfig(n_splits=3))
    assert len(seen) == len(df)
    np.testing.assert_allclose(predictions['utility_score'], df.semantic_rank / 10)
    assert metrics['num_queries'] == 3

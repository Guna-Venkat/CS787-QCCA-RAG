"""Candidate-specific sequential verifier, paper-grouped OOF, preregistered gate."""
import numpy as np
import pandas as pd
from .provenance import assert_inference_clean

FEATURES = ['candidate_query_similarity', 'candidate_draft_similarity', 'candidate_tokens',
            'accepted_count', 'proposed_prompt_tokens']


def headroom_gate(rows, thresholds):
    df = pd.DataFrame(rows)
    fixed = df.groupby('evidence_k').agg(f1=('f1','mean'), cost=('total_query_prompt_tokens','mean'))
    best_count = fixed.sort_values(['f1','cost'], ascending=[False,True]).index[0]
    best = fixed.loc[best_count]
    chosen = []
    for _, g in df.groupby('question_id'):
        if set(g.evidence_k) != {0,1,2,3,4}:
            raise ValueError('Sequential gate needs complete prefix 0/1/2/3/4 outcomes')
        chosen.append(g[g.f1 >= g.f1.max()-thresholds['epsilon']].sort_values('evidence_k').iloc[0])
    oracle = pd.DataFrame(chosen)
    gain = float(oracle.f1.mean()-best.f1)
    savings = float(1-oracle.total_query_prompt_tokens.mean()/best.cost) if best.cost else 0.
    proceed = gain >= thresholds['quality_gain'] or (
        savings >= thresholds['token_savings'] and gain >= -thresholds['quality_tolerance'])
    return dict(status='PROCEED' if proceed else 'SKIPPED — insufficient adaptive headroom',
        proceed=proceed, best_fixed_evidence_k=int(best_count), epsilon_oracle_gain=gain,
        epsilon_oracle_token_savings=savings, thresholds=thresholds)


def online_features(inference, proposed, encoder, accepted_count, proposed_prompt_tokens):
    assert_inference_clean(inference)
    assert_inference_clean(proposed)
    from .retrieval import unit_vectors
    q, d, c = unit_vectors(encoder.encode([inference['question'], inference['initial_draft'], proposed['text']]))
    return dict(candidate_query_similarity=float(c@q), candidate_draft_similarity=float(c@d),
        candidate_tokens=proposed['token_length'], accepted_count=accepted_count,
        proposed_prompt_tokens=proposed_prompt_tokens)


def training_actions(rows):
    """Gold deltas are FIT supervised targets; features saved before evaluation."""
    records = []
    for qid, g in pd.DataFrame(rows).groupby('question_id'):
        g = g.sort_values('evidence_k')
        if g.evidence_k.tolist() != [0,1,2,3,4]:
            raise ValueError('Sequential training requires every prefix, including K_E=3')
        for previous, current in zip(g.to_dict('records'), g.to_dict('records')[1:]):
            if current.get('evidence_actual_count', current['evidence_k']) != current['evidence_k']:
                continue
            if current['selected_chunk_ids'][:-1] != previous['selected_chunk_ids']:
                raise ValueError('Sequential supervision requires nested candidate prefixes')
            feature = current['proposed_features']
            assert_inference_clean(feature)
            if not feature or set(feature) != set(FEATURES):
                raise ValueError('Missing real candidate-specific action features')
            records.append(dict(question_id=qid, paper_id=current['paper_id'],
                evidence_k=current['evidence_k'], candidate_chunk_id=current['selected_chunk_ids'][-1],
                **feature, marginal_target=current['f1']-previous['f1'],
                marginal_cost=current['total_query_prompt_tokens']-previous['total_query_prompt_tokens']))
    return pd.DataFrame(records)


def train_oof(actions, folds=5):
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    x, y = actions[FEATURES].to_numpy(float), actions.marginal_target.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite marginal actions')
    predictions = np.full(len(actions), np.nan)
    fold_ids = np.full(len(actions), -1)
    models = {}
    for i, (tr, va) in enumerate(GroupKFold(folds).split(x, y, actions.paper_id)):
        if set(actions.iloc[tr].paper_id) & set(actions.iloc[va].paper_id):
            raise ValueError('Paper leakage')
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.))
        model.fit(x[tr], y[tr])
        predictions[va] = model.predict(x[va])
        fold_ids[va] = i
        models[i] = model
    final = make_pipeline(StandardScaler(), Ridge(alpha=1.)).fit(x,y)
    return predictions, fold_ids, models, final


def sequential_stop(actions, predicted_utilities, lambda_cost=0.):
    work = actions.assign(predicted_utility=predicted_utilities)
    output = []
    for qid, g in work.groupby('question_id'):
        accepted = []
        attempted = 0
        for row in g.sort_values('evidence_k').itertuples():
            attempted += 1
            # Prefix-only stopping keeps training/online state distributions aligned.
            # On rejection STOP immediately; no unmeasured skipped-prefix counterfactuals.
            if row.predicted_utility-lambda_cost*row.proposed_prompt_tokens <= 0:
                break
            accepted.append(row.candidate_chunk_id)
        output.append(dict(question_id=qid, paper_id=str(g.iloc[0].paper_id),
            evidence_k=len(accepted), attempted_count=attempted, selected_chunk_ids=accepted))
    return output

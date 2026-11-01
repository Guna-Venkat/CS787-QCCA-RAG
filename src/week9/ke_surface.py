"""Retrospective evidence-count oracles; never imported by online selection."""
import pandas as pd


def surface_analysis(rows, counts=(0,1,2,4), epsilon=.01):
    df = pd.DataFrame(rows)
    keys = ['question_id', 'paper_id']
    if df.duplicated(keys + ['evidence_k']).any():
        raise ValueError('Duplicate evidence-count outcomes')
    records = []
    for key, g in df.groupby(keys):
        if set(g.evidence_k) != set(counts):
            raise ValueError('Incomplete K_E response surface')
        best = g.f1.max()
        oracle = g.loc[g.f1 == best].sort_values('evidence_k').iloc[0]
        eps = g.loc[g.f1 >= best-epsilon-1e-12].sort_values('evidence_k').iloc[0]
        row = dict(zip(keys, key))
        row.update(oracle_evidence_k=int(oracle.evidence_k), epsilon_oracle_evidence_k=int(eps.evidence_k))
        for r in g.itertuples():
            row[f'f1_evidence_k_{r.evidence_k}'] = r.f1
        records.append(row)
    matrix = pd.DataFrame(records)
    summary = dict(oracle_distribution=matrix.oracle_evidence_k.value_counts().sort_index().to_dict(),
        epsilon_oracle_distribution=matrix.epsilon_oracle_evidence_k.value_counts().sort_index().to_dict(),
        epsilon=epsilon, oracle_is_deployable=False,
        fixed_evidence_k=df.groupby('evidence_k')[['f1','final_prompt_tokens','total_query_prompt_tokens',
            'total_query_generated_tokens','help','hurt']].mean().reset_index().to_dict('records'))
    return matrix, summary

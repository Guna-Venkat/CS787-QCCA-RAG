"""Fixed-k FIT selection and retrospective oracles; never called by scoring."""
import numpy as np
import pandas as pd
from ..caching import atomic_json
from ..provenance import digest,file_hash
from ..evaluation import paired_paper_bootstrap
from .protocol import GRID

METRICS=('f1','exact_match','delta_f1','help','hurt','answer_changed','final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','retrieval_query_time','selection_latency','extraction_latency','revision_latency','measured_end_to_end_latency',
 'evidence_tokens','actual_sentence_k','mean_selected_cosine','min_selected_cosine','next_unselected_cosine','selected_vs_next_margin','compression_ratio','source_support_validation_rate',
 'same_chunk_pair_fraction','all_selected_same_chunk','adjacent_sentence_pair_rate','mean_pairwise_selected_cosine','exact_duplicate_sentence_pair_rate','near_duplicate_sentence_pair_rate',
 'sentence_index_conceptual_seconds','sentence_index_current_compute_seconds','sentence_embedding_lookup_seconds','query_embedding_conceptual_seconds','query_embedding_current_compute_seconds','query_embedding_lookup_seconds','embedding_lookup_compute_latency','sentence_ranking_latency')


def summaries(rows):
    result=[]
    for policy,g in pd.DataFrame(rows).groupby('policy',sort=False):
        row=dict(policy=policy,n_queries=len(g))
        for key in METRICS:
            row[key]=float(g[key].mean()) if key in g and g[key].notna().any() else None
        if 'sentence_k' in g and g.sentence_k.notna().any():row['sentence_k']=int(g.sentence_k.iloc[0])
        if 'evidence_tokens' in g and g.evidence_tokens.notna().any():
            row.update(median_evidence_tokens=float(g.evidence_tokens.median()),p90_evidence_tokens=float(g.evidence_tokens.quantile(.9)))
        denom=(row.get('help') or 0)+(row.get('hurt') or 0)
        row['descriptive_help_over_help_plus_hurt']=row['help']/denom if denom else None
        result.append(row)
    return result


def choose_fixed(candidates,tolerance=.003):
    if not candidates or not all(np.isfinite(r['f1']) for r in candidates):raise ValueError('Invalid fixed-k summary')
    best=max(r['f1'] for r in candidates)
    eligible=[r for r in candidates if r['f1']>=best-tolerance-1e-12]
    return min(eligible,key=lambda r:(r['total_query_prompt_tokens'],r['final_prompt_tokens'],r['sentence_k']))


def oracle(rows,d0,*,include_zero=False,allowed=None,epsilon=.01):
    grid=list(allowed or GRID)
    grouped={}
    for r in rows:
        if r['sentence_k'] in grid:
            q=r['question_id'];k=r['sentence_k']
            if k in grouped.setdefault(q,{}):raise ValueError('Duplicate oracle action')
            grouped[q][k]=r
    baseline={r['question_id']:r for r in d0}
    choices=[];costs=[]
    for q,actions in sorted(grouped.items()):
        if set(actions)!=set(grid):raise ValueError('Incomplete oracle surface')
        if include_zero:
            if q not in baseline or baseline[q]['paper_id']!=next(iter(actions.values()))['paper_id']:raise ValueError('D0 oracle query/paper mismatch')
            actions={**actions,0:baseline[q]}
        best=max(r['f1'] for r in actions.values())
        exact=min(k for k,r in actions.items() if r['f1']==best)
        eps=min(k for k,r in actions.items() if r['f1']>=best-epsilon-1e-12)
        choices.append(dict(question_id=q,paper_id=actions[exact]['paper_id'],exact_oracle_k=exact,epsilon_oracle_k=eps,exact_oracle_f1=actions[exact]['f1'],epsilon_oracle_f1=actions[eps]['f1']))
        for name,k in [('exact',exact),('epsilon',eps)]:
            r=actions[k];costs.append(dict(question_id=q,oracle=name,k=k,f1=r['f1'],evidence_tokens=r.get('evidence_tokens',0),final_prompt_tokens=r['final_prompt_tokens'],total_query_prompt_tokens=r['total_query_prompt_tokens'],total_query_generated_tokens=r['total_query_generated_tokens'],num_lm_calls=r['num_lm_calls']))
    if include_zero and set(grouped)!=set(baseline):raise ValueError('D0 oracle coverage mismatch')
    df=pd.DataFrame(choices);cost=pd.DataFrame(costs);summary={}
    for label in ('exact','epsilon'):
        g=cost[cost.oracle==label]
        summary[label]=dict(f1=float(g.f1.mean()),mean_k=float(g.k.mean()),median_k=float(g.k.median()),
            mean_evidence_tokens=float(g.evidence_tokens.mean()),mean_final_prompt_tokens=float(g.final_prompt_tokens.mean()),
            mean_cumulative_prompt_tokens=float(g.total_query_prompt_tokens.mean()),mean_cumulative_generated_tokens=float(g.total_query_generated_tokens.mean()),
            mean_lm_calls=float(g.num_lm_calls.mean()),distribution={str(k):int((g.k==k).sum()) for k in ([0]+grid if include_zero else grid)},
            deployable=False,cost_note='Costs of retrospectively chosen actions, excluding the cost of searching the oracle; not inference-time achievable costs.')
    return df,summary,cost


def dynamic_decision(full_oracle,intervention_oracle,candidates,cfg):
    # Use max point-estimate F1 as the conservative benchmark, not a possibly
    # lower-F1 cost-tolerant selected policy that would inflate oracle headroom.
    best=min(candidates,key=lambda r:(-r['f1'],r['total_query_prompt_tokens'],r['final_prompt_tokens'],r['sentence_k']))
    eps=full_oracle['epsilon'];gain=eps['f1']-best['f1']
    reductions={}
    for name,a,b in [('evidence',eps['mean_evidence_tokens'],best['evidence_tokens']),('final_prompt',eps['mean_final_prompt_tokens'],best['final_prompt_tokens']),('cumulative_prompt',eps['mean_cumulative_prompt_tokens'],best['total_query_prompt_tokens'])]:
        reductions[name]=1-a/b if b else None
    c1=gain>=cfg['dynamic_gain']-1e-12
    c2=gain>=-cfg['quality_tolerance']-1e-12 and any(v is not None and v>=cfg['dynamic_cost_saving']-1e-12 for v in reductions.values())
    counts=intervention_oracle['epsilon']['distribution'];total=sum(counts.values())
    masses={k:v/total for k,v in counts.items()} if total else {}
    nondegenerate=sum(v>=cfg['heterogeneity_min_mass'] for v in masses.values())>=2
    changes=max(max(r['help'] for r in candidates)-min(r['help'] for r in candidates),max(r['hurt'] for r in candidates)-min(r['hurt'] for r in candidates))
    c3=nondegenerate and changes>=cfg['heterogeneity_help_hurt_range'] and gain>=cfg['heterogeneity_min_gain']
    return dict(recommendation='GO' if c1 or c2 or c3 else 'NO-GO',controller_trained_or_run=False,
        comparator_best_point_estimate_k=best['sentence_k'],oracle_f1_gain=gain,cost_reductions=reductions,
        gain_criterion=bool(c1),cost_criterion=bool(c2),heterogeneity_criterion=bool(c3),intervention_epsilon_mass=masses,
        help_hurt_rate_range=changes,criterion3_definition=dict(min_mass=cfg['heterogeneity_min_mass'],help_hurt_range=cfg['heterogeneity_help_hurt_range'],min_oracle_gain=cfg['heterogeneity_min_gain']),
        interpretation='Retrospective headroom recommendation only; GO does not establish a deployable dynamic controller or authorize training.')


def analyze(rows,baselines,cfg,ident,directory,complete):
    directory.mkdir(parents=True,exist_ok=True)
    expected={(r['question_id'],r['paper_id'],k) for r in rows for k in GRID}
    actual=[(r['question_id'],r['paper_id'],r['sentence_k']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Incomplete E3 six-k grid')
    table=summaries(baselines+rows);pd.DataFrame(table).to_csv(directory/'summary.csv',index=False)
    candidates=[r for r in table if r['policy'].startswith('E3-k')]
    groups={p:[r for r in rows+baselines if r['policy']==p] for p in {r['policy'] for r in rows+baselines}}
    if not complete:
        (directory/'REPORT.md').write_text('Tiny deterministic FIT smoke only. No k selection, oracle-driven method tuning, freeze or dynamic-controller work. Exact source matching does not imply relevance or answer correctness.\n')
        return None
    selected=choose_fixed(candidates,cfg['quality_tolerance'])
    comparisons={}
    for k in GRID:
        for base in ('D0','D1'):comparisons[f'E3-k{k} - {base}']=paired_paper_bootstrap(groups[f'E3-k{k}'],groups[base],n=10000,seed=42)
    for base in ('D0','D1','E2'):comparisons[f'best-fixed - {base}']=paired_paper_bootstrap(groups[selected['policy']],groups[base],n=10000,seed=42)
    pd.DataFrame([dict(comparison=k,point_estimate=v['point_estimate'],ci95_low=v['ci95'][0],ci95_high=v['ci95'][1],resamples=10000,cluster='paper_id',seed=42) for k,v in comparisons.items()]).to_csv(directory/'bootstrap.csv',index=False)
    atomic_json(directory/'bootstrap.json',comparisons)
    oracle_summaries={};distributions=[]
    d0=groups['D0']
    for name,zero,allowed in [('intervention_only',False,GRID),('full_action',True,GRID),('local_window_intervention_only',False,[k for k in GRID if abs(k-selected['sentence_k'])<=3])]:
        choices,summary,cost=oracle(rows,d0,include_zero=zero,allowed=allowed,epsilon=cfg['epsilon'])
        choices.to_csv(directory/f'{name}_oracle_choices.csv',index=False);cost.to_csv(directory/f'{name}_oracle_costs.csv',index=False)
        oracle_summaries[name]=summary
        for variant,g in summary.items():
            for k,count in g['distribution'].items():distributions.append(dict(analysis=name,oracle=variant,k=int(k),queries=count,rate=count/len(choices)))
    pd.DataFrame(distributions).to_csv(directory/'oracle_distribution.csv',index=False)
    atomic_json(directory/'oracle_summary.json',oracle_summaries)
    decision=dynamic_decision(oracle_summaries['full_action'],oracle_summaries['intervention_only'],candidates,cfg)
    atomic_json(directory/'dynamic_go_no_go.json',decision)
    notes=[]
    for k in GRID:
        raw=comparisons[f'E3-k{k} - D1'];base=comparisons[f'E3-k{k} - D0']
        if raw['ci95'][0]>0:notes.append(f'k={k}: sentence suppression improves over raw D1 under this frozen selector on FIT (unadjusted paired interval).')
        if base['ci95'][0]>0:notes.append(f'k={k}: sentence intervention improves over D0 on FIT (unadjusted paired interval).')
    if not notes:notes.append('No fixed-k superiority is established by positive paired intervals. Query-sentence similarity may be insufficient; inconclusive intervals do not prove equivalence.')
    notes.append('Fixed-k selection and its best-policy intervals use the same FIT data: post-selection descriptive evidence, not independent confirmation. Multiple fixed-k intervals are unadjusted. High cosine is neither causal evidence nor factual entailment. No MMR or dynamic controller was run.')
    notes.append('Local window uses only evaluated k values with |k-k0|<=3; no untested k=5/7 is fabricated. Gold sentence overlap is omitted because no validated exact sentence-level mapping was prepared.')
    (directory/'REPORT.md').write_text('# E3-SentTopK FIT diagnostic\n\n'+'\n\n'.join(notes)+f"\n\nDynamic-k recommendation: {decision['recommendation']}; no controller trained or run. Primary policy remains K=0.\n")
    freeze=dict(schema='E3-SentTopK-v1',frozen=True,selection_split='fit',selected_k=selected['sentence_k'],selection_rule=ident['selection_rule'],all_candidates=candidates,
        protocol_identity_hash=digest(ident),protocol=ident,source_identity=ident['input_hashes'],
        artifact_hashes={str(p.resolve()):file_hash(p) for p in [directory/'summary.csv',directory/'bootstrap.csv',directory/'oracle_summary.json',directory/'dynamic_go_no_go.json',directory/'results.jsonl']},
        primary_policy_unchanged='K_E=0',diagnostic_only=True)
    path=directory.parent/'frozen_e3_senttopk.json'
    if path.exists():
        import json
        if json.loads(path.read_text())!=freeze:raise ValueError('Refusing to overwrite a different fixed-k E3 freeze')
    else:atomic_json(path,freeze)
    return dict(selected=selected,dynamic=decision)

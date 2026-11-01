"""Saved FIT action ceilings, deterministic ties and explicit research screens."""
import math
import json
from itertools import combinations
import numpy as np
import pandas as pd
from ..provenance import digest,file_hash
from ..caching import atomic_json

COSTS=('final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls','evidence_tokens')
SPACES={'PRE-A':('D0','F2-A'),'PRE-B':('D0','F0'),'PRE-C':('D0','F0','F2-A')}


def average(values):
    return float(np.mean(values)) if values else None


def state(delta,epsilon=.01):
    return 'HELP' if delta>epsilon else 'HURT' if delta< -epsilon else 'NEUTRAL'


def target(d0,f0,epsilon=.01):
    delta=f0-d0
    return dict(delta_action=delta,accept_f0=int(delta>epsilon),action_state=state(delta,epsilon))


def unique_best(actions):
    best=max(actions.values());tied=[p for p,s in actions.items() if s==best]
    if len(tied)==1:return tied[0]+' uniquely best'
    if len(tied)==3:return 'three-way tie'
    for a,b in (('D0','F0'),('D0','F2-A'),('F0','F2-A')):
        if set(tied)=={a,b}:return f'tie involving {a}/{b}'
    raise ValueError('Unexpected action set')


def oracle(groups,actions,preference=('D0','F2-A','F0'),epsilon=.01):
    keys=set(groups['D0'])
    if any(set(groups[p])!=keys for p in actions):raise ValueError('Missing/misaligned oracle action')
    ordering=[p for p in preference if p in actions];choices=[]
    for q in sorted(keys):
        rows={p:groups[p][q] for p in actions};base=groups['D0'][q]
        if any(r['paper_id']!=base['paper_id'] for r in rows.values()):raise ValueError('Cross-paper oracle action')
        best=max(r['f1'] for r in rows.values())
        exact=next(p for p in ordering if rows[p]['f1']==best)
        eps=next(p for p in ordering if rows[p]['f1']>=best-epsilon)
        for kind,p in [('exact',exact),('epsilon',eps)]:
            r=rows[p]
            choices.append(dict(question_id=q,paper_id=r['paper_id'],oracle=kind,action=p,
                f1=r['f1'],exact_match=r['exact_match'],delta_vs_d0=r['f1']-base['f1'],
                selected_state=state(r['f1']-base['f1'],epsilon),
                **{k:r[k] for k in COSTS}))
    best_fixed=max(average([r['f1'] for r in groups[p].values()]) for p in actions)
    summary=[];distribution=[]
    for kind in ('exact','epsilon'):
        rows=[r for r in choices if r['oracle']==kind]
        summary.append(dict(oracle=kind,n_queries=len(rows),f1=average([r['f1'] for r in rows]),
            exact_match=average([r['exact_match'] for r in rows]),best_fixed_f1=best_fixed,
            headroom=average([r['f1'] for r in rows])-best_fixed,
            help=average([r['selected_state']=='HELP' for r in rows]),
            hurt=average([r['selected_state']=='HURT' for r in rows]),
            **{k:average([r[k] for r in rows]) for k in COSTS}))
        for p in actions:
            chosen=[r for r in rows if r['action']==p]
            distribution.append(dict(oracle=kind,action=p,count=len(chosen),rate=len(chosen)/len(rows),
                help_count=sum(r['selected_state']=='HELP' for r in chosen),
                hurt_count=sum(r['selected_state']=='HURT' for r in chosen),
                neutral_count=sum(r['selected_state']=='NEUTRAL' for r in chosen)))
    return summary,distribution,choices


def screens(metrics,unique,cfg):
    a=metrics['PRE-A']['epsilon']['headroom'];b=metrics['PRE-B']['epsilon']['headroom']
    extra=metrics['PRE-C']['exact']['f1']-metrics['PRE-B']['exact']['f1']
    advantage=b-a
    g1=a>=cfg['minimum_gate_headroom']
    g2=b>=cfg['minimum_gate_headroom'] and advantage>=cfg['minimum_verifier_advantage']
    g3=extra>=cfg['minimum_three_action_gain']
    # Three-way is never a default. It requires its own additional-value screen.
    recommended='G3' if g3 else 'G2' if g2 else 'G1' if g1 else 'NO-GO'
    return dict(G1=dict(decision='GO' if g1 else 'NO-GO',epsilon_headroom=a,threshold=cfg['minimum_gate_headroom'],
                    rule='epsilon oracle(D0,F2-A) minus best fixed >= threshold'),
        G2=dict(decision='GO' if g2 else 'NO-GO',epsilon_headroom=b,verifier_advantage=advantage,
                threshold=cfg['minimum_gate_headroom'],material_advantage_threshold=cfg['minimum_verifier_advantage'],
                rule='epsilon oracle(D0,F0) headroom >=.005 AND advantage over PRE-A headroom >=.005; descriptive screen, not significance'),
        G3=dict(decision='GO' if g3 else 'NO-GO',extra_exact_oracle_f1=extra,
                unique_best_rate=unique['f2_unique_exact_best_rate'],threshold=cfg['minimum_three_action_gain'],
                rule='Three-action exact oracle minus D0/F0 exact oracle >=.003. Unique-best fraction is descriptive; no outcome-tuned extra cutoff.'),
        recommended_route=recommended,controller_training_started=False,lm_calls=0,
        interpretation='Retrospective action-space research recommendation, not evidence of a learnable/deployable verifier.')


def analyze(groups,cfg,directory):
    values=[];all_summaries=[];all_dist=[];all_choices=[];metrics={}
    for p,g in groups.items():
        values.append(dict(action=p,n_queries=len(g),f1=average([r['f1'] for r in g.values()]),
            exact_match=average([r['exact_match'] for r in g.values()]),
            help=average([state(r['f1']-groups['D0'][q]['f1'],cfg['epsilon'])=='HELP' for q,r in g.items()]),
            hurt=average([state(r['f1']-groups['D0'][q]['f1'],cfg['epsilon'])=='HURT' for q,r in g.items()]),
            **{k:average([r[k] for r in g.values()]) for k in COSTS}))
    for name,actions in SPACES.items():
        summary,dist,choices=oracle(groups,actions,cfg['action_preference'],cfg['epsilon'])
        metrics[name]={r['oracle']:r for r in summary}
        all_summaries.extend(dict(action_space=name,**r) for r in summary)
        all_dist.extend(dict(action_space=name,**r) for r in dist)
        all_choices.extend(dict(action_space=name,**r) for r in choices)
    counts={k:0 for k in ('D0 uniquely best','F0 uniquely best','F2-A uniquely best',
        'tie involving D0/F0','tie involving D0/F2-A','tie involving F0/F2-A','three-way tie')}
    records=[];labels=[];f2_only_eps=0;strictsame=0;normsame=0
    def norm(s):return ' '.join(s.lower().split())
    for q in sorted(groups['D0']):
        a={p:groups[p][q]['f1'] for p in ('D0','F0','F2-A')}
        category=unique_best(a);counts[category]+=1
        f2_only_eps+=a['F2-A']>=max(a.values())-cfg['epsilon'] and all(a[p]<max(a.values())-cfg['epsilon'] for p in ('D0','F0'))
        strictsame+=groups['F2-A'][q]['prediction'].strip()==groups['D0'][q]['prediction'].strip()
        normsame+=norm(groups['F2-A'][q]['prediction'])==norm(groups['D0'][q]['prediction'])
        records.append(dict(question_id=q,paper_id=groups['D0'][q]['paper_id'],category=category,**a))
        labels.append(dict(question_id=q,paper_id=groups['D0'][q]['paper_id'],
            **target(a['D0'],a['F0'],cfg['epsilon'])))
    n=len(groups['D0'])
    unique=dict(f2_unique_exact_best_count=counts['F2-A uniquely best'],f2_unique_exact_best_rate=counts['F2-A uniquely best']/n,
        f2_only_epsilon_eligible_count=int(f2_only_eps),f2_only_epsilon_eligible_rate=f2_only_eps/n,
        f2_same_strict_answer_as_d0_rate=strictsame/n,f2_same_case_whitespace_normalized_answer_as_d0_rate=normsame/n,
        f2_equal_f1_to_d0_rate=average([groups['F2-A'][q]['f1']==groups['D0'][q]['f1'] for q in groups['D0']]),
        exact_extra_oracle_f1=metrics['PRE-C']['exact']['f1']-metrics['PRE-B']['exact']['f1'],
        epsilon_extra_oracle_f1=metrics['PRE-C']['epsilon']['f1']-metrics['PRE-B']['epsilon']['f1'])
    balance={s:sum(r['action_state']==s for r in labels) for s in ('HELP','NEUTRAL','HURT')}
    label_summary=dict(n_queries=n,counts=balance,rates={k:v/n for k,v in balance.items()},
        accept_target_prevalence=balance['HELP']/n,definition='FIT target for THIS F0: delta=F0_F1-D0_F1; ACCEPT iff delta>.01; continuous delta retained. Labels never inference features.')
    decision=screens(metrics,unique,cfg)
    headroom=dict(action_spaces=metrics,f2_unique_contribution=unique,action_label_balance=label_summary,
        verifier_advantage=metrics['PRE-B']['epsilon']['headroom']-metrics['PRE-A']['epsilon']['headroom'])
    # Chosen-action accounting is retrospective. Verification has sunk candidate cost.
    candidate_costs=[]
    for name,actions in SPACES.items():
        generated=[p for p in actions if p!='D0']
        row=dict(action_space=name,note='Generate all non-D0 candidates before action verification; not oracle chosen-action cost.')
        for k in ('total_query_prompt_tokens','total_query_generated_tokens','num_lm_calls'):
            row[k]=average([sum(groups[p][q][k] for p in generated)-(len(generated)-1)*groups['D0'][q][k] for q in groups['D0']])
        candidate_costs.append(row)
    directory.mkdir(parents=True,exist_ok=True)
    tables={'action_values.csv':values,'oracle_summary.csv':all_summaries,'oracle_distribution.csv':all_dist,
        'oracle_choices.csv':all_choices,'three_action_unique_best.csv':[dict(category=k,count=v,rate=v/n) for k,v in counts.items()],
        'three_action_query_categories.csv':records,'action_targets_fit_only.csv':labels,
        'post_generation_candidate_costs.csv':candidate_costs}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(directory/name,index=False)
    atomic_json(directory/'headroom_analysis.json',headroom)
    decision['artifact_hashes']={str((directory/name).resolve()):file_hash(directory/name) for name in list(tables)+['headroom_analysis.json']}
    atomic_json(directory/'go_no_go.json',decision)
    atomic_json(directory/'w9g_preflight_decision.json',decision)
    # Compact standalone report, derived only from saved outcomes.
    def table(rows):return pd.DataFrame(rows).to_markdown(index=False,floatfmt='.6f')
    text=('# W9-G-PRE — Saved-output action-space preflight\n\n'
        'FIT only. Zero model generation and zero controller training. Oracles use observed FIT scores and are not deployable.\n\n'
        '## Fixed saved actions\n\n'+table(values)+'\n\n## Oracle spaces\n\n'+table(all_summaries)+
        '\n\nExact/epsilon preference: D0, then F2-A, then F0; epsilon=0.01. Epsilon chooses the first eligible action, not necessarily the highest score.\n\n'
        '## Action distribution\n\n'+table(all_dist)+'\n\n## Unique exact-best cases\n\n'+table(tables['three_action_unique_best.csv'])+
        '\n\n## F2-A contribution\n\n```json\n'+json.dumps(unique,indent=2)+'\n```\n\n## Research route screens\n\n```json\n'+
        json.dumps({k:v for k,v in decision.items() if k!='artifact_hashes'},indent=2)+'\n```\n\n'
        '## Action targets\n\n```json\n'+json.dumps(label_summary,indent=2)+'\n```\n\n'
        '## Cost limitation\n\nOracle costs count the chosen action only and exclude action search. A post-generation F0 verifier pays the F0 candidate cost even when it keeps D0. '
        'It is a quality/safety mechanism, not a compute-saving retrieval gate. Candidate-generation costs are recorded separately. No measured end-to-end latency is invented.\n\n'
        'A high oracle is an optimistic ceiling; it does not establish that inference-available features can distinguish HELP from HURT. FIT was already used for E3/F development; '
        'later paper-grouped OOF evaluates the verifier conditionally on those frozen upstream actions, not an independent end-to-end confirmation. No DEV/INTERNAL-VAL/TEST data is used.\n')
    (directory/'REPORT.md').write_text(text)
    return decision

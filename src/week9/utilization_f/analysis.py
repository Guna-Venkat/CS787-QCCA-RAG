"""FIT-only utilization metrics, paper bootstrap and immutable action freezes."""
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
from ..evaluation import paired_paper_bootstrap
from ..provenance import digest, file_hash
from .prompts import POLICIES, F2_VARIANTS, definitions
from .protocol import pin, SELECTION_RULE


def normalized(text):
    return ' '.join(re.sub(r'[^\w\s]','',text.lower()).split())


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def lexical(answer,draft,evidence):
    a=set(normalized(answer).split());d=set(normalized(draft).split());e=set(normalized(evidence).split())
    added=a-d
    return dict(answer_evidence_token_overlap=len(a&e)/len(a) if a else None,
                added_answer_token_count=len(added),removed_answer_token_count=len(d-a),
                added_tokens_in_evidence=len(added&e),
                added_token_evidence_fraction=len(added&e)/len(added) if added else None)


def behavior(rows,f0):
    if not rows: raise ValueError('Missing utilization outcomes')
    hurt=[r for r in rows if f0[r['question_id']]['delta_f1'] < -.01]
    helped=[r for r in rows if f0[r['question_id']]['delta_f1'] > .01]
    def rate(group,fn):return mean([bool(fn(r,f0[r['question_id']])) for r in group])
    return dict(f0_hurt_queries=len(hurt),f0_help_queries=len(helped),
        partial_rescue_rate=rate(hurt,lambda r,b:r['f1']>b['f1']+.01),
        full_recovery_rate=rate(hurt,lambda r,b:r['f1']>=r['initial_f1']-.01),
        hurt_rescue_rate=rate(hurt,lambda r,b:not r['hurt']),
        help_retention_rate=rate(helped,lambda r,b:r['help']),
        lost_gain_rate=rate(helped,lambda r,b:not r['help']),
        diminished_gain_rate=rate(helped,lambda r,b:r['f1']<b['f1']-.01))


COST_FIELDS=('final_prompt_tokens','total_query_prompt_tokens','total_query_generated_tokens',
             'num_lm_calls','evidence_tokens','retrieval_query_time','selection_latency',
             'revision_latency','measured_end_to_end_latency','inherited_sentence_index_conceptual_seconds',
             'inherited_query_embedding_conceptual_seconds','inherited_embedding_lookup_compute_latency',
             'inherited_sentence_ranking_latency')


def summarize(rows):
    grouped={p:[r for r in rows if r['policy']==p] for p in ('D0',)+POLICIES}
    f0={r['question_id']:r for r in grouped['F0']}
    results=[];transitions=[];rescues=[]
    for policy,g in grouped.items():
        if not g:raise ValueError(f'Missing policy: {policy}')
        summary=dict(policy=policy,n_queries=len(g),f1=mean([r['f1'] for r in g]),
            exact_match=mean([r['exact_match'] for r in g]),delta_f1=mean([r['delta_f1'] for r in g]),
            delta_f1_vs_f0=mean([r['f1']-f0[r['question_id']]['f1'] for r in g]),
            help=mean([r['help'] for r in g]),hurt=mean([r['hurt'] for r in g]),
            answer_changed=mean([r['answer_changed'] for r in g]),
            strict_preservation_rate=mean([r['prediction'].strip()==r['initial_draft'].strip() for r in g]),
            normalized_preservation_rate=mean([normalized(r['prediction'])==normalized(r['initial_draft']) for r in g]),
            normalized_intervention_rate=mean([normalized(r['prediction'])!=normalized(r['initial_draft']) for r in g]),
            answer_changed_vs_f0=mean([normalized(r['prediction'])!=normalized(f0[r['question_id']]['prediction']) for r in g]),
            f1_differs_vs_f0=mean([abs(r['f1']-f0[r['question_id']]['f1'])>1e-12 for r in g]),
            mean_delta_among_help=mean([r['delta_f1'] for r in g if r['help']]),
            mean_delta_among_hurt=mean([r['delta_f1'] for r in g if r['hurt']]),
            answer_fallback_rate=mean([r.get('answer_fail_closed',False) for r in g]),
            **{k:mean([r.get(k) for r in g]) for k in COST_FIELDS})
        denom=summary['help']+summary['hurt']
        summary['descriptive_help_over_help_plus_hurt']=summary['help']/denom if denom else None
        summary['hurt_reduction_vs_f0']=mean([r['hurt'] for r in grouped['F0']])-summary['hurt']
        summary['help_reduction_vs_f0']=mean([r['help'] for r in grouped['F0']])-summary['help']
        summary.update(behavior(g,f0));results.append(summary)
        rescues.append(dict(policy=policy,**behavior(g,f0)))
        for state in ('HELP','HURT','UNCHANGED'):
            subset=[r for r in g if ('HELP' if f0[r['question_id']]['help'] else 'HURT' if f0[r['question_id']]['hurt'] else 'UNCHANGED')==state]
            transitions.append(dict(policy=policy,f0_state=state,n_queries=len(subset),
                mean_delta_vs_d0=mean([r['delta_f1'] for r in subset]),
                mean_delta_vs_f0=mean([r['f1']-f0[r['question_id']]['f1'] for r in subset]),
                preserve_rate=mean([normalized(r['prediction'])==normalized(r['initial_draft']) for r in subset]),
                help_rate=mean([r['help'] for r in subset]),hurt_rate=mean([r['hurt'] for r in subset]),
                **(behavior(subset,f0) if subset else {})))
    return results,transitions,rescues


def choose(candidates,cfg):
    if not candidates or any(r['policy']=='D0' or not math.isfinite(r['f1']) for r in candidates):
        raise ValueError('Only measured intervention candidates are eligible')
    best=max(r['f1'] for r in candidates)
    eligible=[r for r in candidates if r['f1']>=best-cfg['quality_tolerance']-1e-12]
    minimum_hurt=min(r['hurt'] for r in eligible)
    prompts=definitions()
    def order(r):
        latency=r.get('revision_latency')
        return (r['total_query_prompt_tokens'],
                int(r['hurt']>=minimum_hurt+cfg['material_hurt_difference']-1e-12),
                r['total_query_generated_tokens'],latency if latency is not None else float('inf'),
                len(prompts[r['policy']]['messages'][0]['content']),r['policy'])
    return min(eligible,key=order)


def validate_rows(rows,records,bundle):
    expected={(r['inference']['question_id'],r['inference']['paper_id'],p) for r in records for p in ('D0',)+POLICIES}
    actual=[(r['question_id'],r['paper_id'],r['policy']) for r in rows]
    if set(actual)!=expected or len(actual)!=len(expected):raise ValueError('Incomplete/duplicate W9-F query/policy grid')
    for r in rows:
        src=bundle['e3_rows'][r['question_id']]
        if r['initial_draft']!=src['initial_draft']:raise ValueError('W9-F D0 draft changed')
        if r['policy']!='D0':
            if (r['evidence_hash']!=src['evidence_hash'] or r['evidence_text']!=src['evidence_text'] or
                    r['selected_sentence_ids']!=src['selected_sentence_ids'] or
                    r['sentence_text_hashes']!=[s['text_hash'] for s in src['selected_sentences']] or
                    r['selected_chunk_ids']!=src['selected_chunk_ids'] or
                    r['source_chunk_hashes']!=src['source_chunk_hashes']):
                raise ValueError('F0/F1/F2 evidence differs from frozen E3-k3')
        elif r['prediction']!=src['initial_draft'] or r['num_lm_calls']!=1:
            raise ValueError('D0 must be skip, not regeneration')


def analyze(rows,cfg,ident,directory,complete):
    table,transitions,rescues=summarize(rows)
    pd.DataFrame(table).to_csv(directory/'summary.csv',index=False)
    pd.DataFrame(transitions).to_csv(directory/'transition_analysis.csv',index=False)
    pd.DataFrame(rescues).to_csv(directory/'rescue_analysis.csv',index=False)
    pd.DataFrame([{k:r.get(k) for k in ('policy','n_queries')+COST_FIELDS} for r in table]).to_csv(directory/'cost_summary.csv',index=False)
    pd.DataFrame([dict(question_id=r['question_id'],paper_id=r['paper_id'],policy=r['policy'],
        **lexical(r['prediction'],r['initial_draft'],r.get('evidence_text',''))) for r in rows]).to_csv(directory/'lexical_diagnostics.csv',index=False)
    if not complete:
        (directory/'REPORT.md').write_text('Tiny deterministic FIT smoke only; no mode/prompt selection, freeze, bootstrap claims, or W9-G training.\n')
        return None
    f2=choose([r for r in table if r['policy'] in F2_VARIANTS],cfg)
    chosen=choose([r for r in table if r['policy'] in ('F0','F1',f2['policy'])],cfg)
    groups={p:[r for r in rows if r['policy']==p] for p in ('D0',)+POLICIES}
    comparisons=[]
    pairs=[('F0','D0'),('F1','D0'),('F1','F0')]
    pairs += [(p,b) for p in F2_VARIANTS for b in ('D0','F0','F1')]
    pairs += [('F2-frozen',b) for b in ('D0','F0','F1')]
    for left,right in pairs:
        actual=f2['policy'] if left=='F2-frozen' else left
        stats=paired_paper_bootstrap(groups[actual],groups[right],n=cfg['bootstrap_resamples'],seed=cfg['seed'])
        comparisons.append(dict(comparison=f'{left} - {right}',actual_left=actual,
            point_estimate=stats['point_estimate'],ci95_low=stats['ci95'][0],ci95_high=stats['ci95'][1],
            resamples=cfg['bootstrap_resamples'],cluster='paper_id',seed=cfg['seed'],
            post_selection=left=='F2-frozen',multiplicity_adjusted=False))
    pd.DataFrame(comparisons).to_csv(directory/'bootstrap.csv',index=False)
    frozen_alias=dict(f2,policy='F2-frozen',actual_policy=f2['policy'],post_selection=True)
    pd.DataFrame(table+[frozen_alias]).to_csv(directory/'summary.csv',index=False)
    prompt=definitions()[f2['policy']]
    f2freeze=dict(schema='W9-F-F2-v1',frozen=True,split='fit',selected_variant=f2['policy'],
                  prompt=prompt,selection_rule=SELECTION_RULE,all_fit_candidates=[r for r in table if r['policy'] in F2_VARIANTS],
                  protocol_hash=digest(ident),not_independent_confirmation=True)
    actionfreeze=dict(schema='W9-F-intervention-v1',frozen=True,split='fit',
        selected_intervention_mode='F2' if chosen['policy'] in F2_VARIANTS else chosen['policy'],
        actual_policy=chosen['policy'],selected_f2_variant=f2['policy'],
        evidence_policy='E3-k3',selection_rule=SELECTION_RULE,all_fit_candidates=[r for r in table if r['policy'] in ('F0','F1',f2['policy'])],
        prompt=definitions()[chosen['policy']],fit_metrics=chosen,protocol_hash=digest(ident),
        generator_identity=ident['model'],source_e3_protocol_hash=ident['upstream_protocol_hash'],
        source_e3_manifest_hash=ident['upstream_manifest_hash'],
        config_hash=digest(ident['config']),skip_action='D0',controller_trained_or_run=False,
        artifact_hashes={str(p.resolve()):file_hash(p) for p in (directory/'results.jsonl',directory/'summary.csv',directory/'bootstrap.csv')})
    pin(directory.parent/'frozen_w9f_f2.json',f2freeze)
    pin(directory.parent/'frozen_w9f_intervention.json',actionfreeze)
    handoff=dict(schema='W9-F-controller-handoff-v1',split='fit',fit_research_labels_only=True,
        not_inference_features=['d0_f1','intervention_f1','delta_f1','help','hurt'],
        controller_trained_or_run=False,intervention_freeze=actionfreeze,
        reason_g_waits='E3 oracle used F0; W9-F can change the action utility surface. G must consume this frozen mode, not hard-coded F1/G1.',
        queries=[dict(question_id=r['question_id'],paper_id=r['paper_id'],d0_f1=r['initial_f1'],
            intervention_f1=r['f1'],delta_f1=r['delta_f1'],help=r['help'],hurt=r['hurt'],
            evidence_hash=r['evidence_hash'],selected_sentence_ids=r['selected_sentence_ids'],
            costs={k:r.get(k) for k in COST_FIELDS}) for r in groups[chosen['policy']]])
    pin(directory.parent/'w9f_controller_handoff.json',handoff)
    lines=['# W9-F FIT utilization diagnostic',f'F2 prompt: {f2["policy"]}; intervention: {chosen["policy"]}. D0 remains external SKIP.',
        'All prompts were predeclared. Selection and selected-policy CIs use FIT: post-selection descriptive evidence, not held-out confirmation.',
        'Intervals are unadjusted paired paper-cluster bootstrap; crossing zero does not establish superiority or equivalence.',
        'Conservative preservation, rescue and lost-gain diagnostics explain the tradeoff; lexical overlap is not causal attribution.',
        'Stage latency excludes cached first-pass and preprocessing; no full end-to-end latency is claimed.',
        'W9-G was not trained/run. No retrieval, sentence ranking, k change or new generator was used.']
    (directory/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    return dict(f2=f2['policy'],intervention=chosen['policy'])

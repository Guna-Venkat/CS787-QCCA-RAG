"""W8-I measured controlled replays plus matched W8-H generator comparisons."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.week8.data import PROJECT_ROOT,load_config
from src.week8.generator_features import canonical_k4_context,sha256_file
from src.week8.qcla_experiment import load_inputs,atomic_json
from src.week8.outcome_weighted_training import frozen_policy
from src.week8.full_system import load_g_variants,sources as h_sources
from src.week8.generation import promoted_prompt
from src.week8.selective_controller import make_decisions
from src.week8.final_reporting import report_systems,paired_interval

OUT=PROJECT_ROOT/'results/week8/w8i_ablations'
H=PROJECT_ROOT/'results/week8/w8h_full_system/full'


def verified_h():
    summary=json.loads((H/'summary.json').read_text())
    _,_,policy_path=frozen_policy()
    adapters=load_g_variants(True,policy_path)
    fingerprint,_=h_sources(summary['settings'],policy_path,adapters)
    if summary['fingerprint']!=fingerprint or summary['smoke_only'] or summary['TEST_used']:raise ValueError('W8-H is not a complete unchanged real run')
    for name,digest in summary['artifact_sha256'].items():
        if sha256_file(H/name)!=digest:raise ValueError('W8-H artifact integrity failed: '+name)
    return summary


def run_w8i(output_dir=OUT,*,check=False,smoke=False):
    verified_h()
    part=load_inputs()['internal_val'];records={str(r['question_id']):r for r in part['records']}
    if check:
        print('W8-I verified: 342 real INTERNAL-VAL queries, completed W8-H, all G adapters; no GPU generation needed')
        return
    folder=Path(output_dir)/('smoke' if smoke else 'full');folder.mkdir(parents=True,exist_ok=True)
    bases={r['question_id']:r for r in (json.loads(l) for l in (H/'live_drafts_and_decisions.jsonl').read_text().splitlines())}
    if set(bases)!=set(records):raise ValueError('W8-H base coverage differs from INTERNAL-VAL')
    wanted=set(sorted(records)[:5] if smoke else records)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(PROJECT_ROOT/load_config()['sara']['checkpoint_path']),local_files_only=True)
    config=json.loads((PROJECT_ROOT/'results/week8/w8e_controller/frozen_controller.json').read_text())['selected_config']
    specs=[('frozen_controller',{}),('no_uncertainty_gate',{'gate_threshold':0.}),
        ('no_hurt_filter',{'tau_hurt':1.01}),('no_help_filter',{'tau_help':-1.}),
        ('always_draftsemantic',{'family':'E1'}),('always_rank5',{'family':'E1','selection':'rank5'}),
        ('always_random_seed42',{'family':'E1','selection':'random'}),
        ('query_semantic_selector',{'selection':'semantic'})]
    systems={'semantic_k4':[]};rng=np.random.default_rng(42)
    sources={'results/week8/w8h_full_system/full/summary.json':sha256_file(H/'summary.json')}
    for name in ['src/week8/ablation_reporting.py','src/week8/final_reporting.py','scripts/week8/run_w8i_ablation_matrix.py']:
        sources[name]=sha256_file(PROJECT_ROOT/name)
    atomic_json(folder/'protocol.json',{'split':'previously inspected INTERNAL-VAL','smoke_only':smoke,
        'specifications':specs,'seed':42,'no_tuning':True,'TEST_used':False,'sources':sources,
        'measurement':'Original-SARA G0 counterfactual replay uses real saved promotion answers. W8-H G0/G1/adapters use actual generated answers. These are separate evidence types.',
        'limitations':'No retraining ablation is claimed. DAFiR-2 and QCLA/controller stacking are unsupported. Removing a safety filter is distinct from retraining without a loss penalty.'})
    for q in sorted(wanted):
        base=bases[q];record=records[q];group=part['routing'][q];emb=part['sfr'][q]
        _,base_prompt,_,_,_,_=canonical_k4_context(group,record,emb,tokenizer)
        base_tokens=len(tokenizer(base_prompt)['input_ids'])
        base_f1=float(group.iloc[0].base_f1)
        systems['semantic_k4'].append({'question_id':q,'paper_id':str(record['paper_id']),'token_f1':base_f1,
            'final_prompt_tokens':base_tokens,'cumulative_prompt_tokens':base_tokens})
        candidates=pd.DataFrame(base['inference']['candidate_predictions'])
        random_rank=int(rng.integers(5,11))
        for name,override in specs:
            cfg={**config,**{k:v for k,v in override.items() if k!='selection'}}
            table=candidates.copy();selection=override.get('selection')
            if selection=='rank5':table['selection_score']=-table.semantic_rank
            elif selection=='random':table['selection_score']=(table.semantic_rank==random_rank).astype(float)
            elif selection=='semantic':table['selection_score']=-table.semantic_rank
            decision=make_decisions(table,{q:base['inference']['gate_probability']},cfg).iloc[0].to_dict()
            promote=decision['selected_action']=='PROMOTE'
            f1=base_f1;tokens=base_tokens
            if promote:
                selected=group[group.candidate_passage_index.astype(int)==int(decision['candidate_passage_index'])]
                if len(selected)!=1:raise ValueError('Counterfactual identity mismatch')
                f1=float(selected.iloc[0].promoted_f1)
                prompt,_,_,_=promoted_prompt(record,group,emb,tokenizer,decision,'G0',base['draft']['prediction'])
                tokens=len(tokenizer(prompt)['input_ids'])
            systems.setdefault('replay/'+name,[]).append({'question_id':q,'paper_id':str(record['paper_id']),
                'token_f1':f1,'selected_action':decision['selected_action'],'final_prompt_tokens':tokens,
                'cumulative_prompt_tokens':base_tokens+(tokens if promote else 0),
                'evidence_type':'measured original-SARA counterfactual replay'})
    for variant in ['original_SARA','uniform_SFT','HC_LoRA','OW_LoRA']:
        rows=[json.loads(l) for l in (H/f'{variant}_predictions.jsonl').read_text().splitlines()]
        for mode in ['G0','G1']:
            chosen=[r for r in rows if r['revision_mode']==mode and r['question_id'] in wanted]
            if len(chosen)!=len(wanted) or len({r['question_id'] for r in chosen})!=len(wanted):raise ValueError('Incomplete W8-H paired results')
            systems[variant+'/'+mode]=chosen
    frames={k:pd.DataFrame(v) for k,v in systems.items()}
    # Replay must reproduce the measured live frozen G0 exactly, including harms.
    a=frames['replay/frozen_controller'].set_index('question_id').sort_index()
    b=frames['original_SARA/G0'].set_index('question_id').sort_index()
    for column in ['token_f1','final_prompt_tokens','cumulative_prompt_tokens']:
        if not np.allclose(a[column],b[column],atol=1e-8):raise ValueError('Frozen replay differs from real W8-H: '+column)
    table=report_systems(frames,folder,'INTERNAL-VAL development ablations; fixed comparisons, no tuning')
    intervals=pd.read_csv(folder/'paired_paper_intervals.csv')
    matched=[]
    for name,frame in frames.items():
        if name.startswith('replay/') and name!='replay/frozen_controller':
            matched.append({'treatment':name,'control':'replay/frozen_controller',**paired_interval(frame,frames['replay/frozen_controller'])})
    pd.concat([intervals,pd.DataFrame(matched)],ignore_index=True).to_csv(folder/'paired_paper_intervals.csv',index=False)
    pd.concat([f.assign(system=k) for k,f in frames.items()],ignore_index=True).to_json(folder/'query_results.jsonl',orient='records',lines=True)
    atomic_json(folder/'summary.json',{'queries':len(wanted),'smoke_only':smoke,'TEST_used':False,
        'artifacts':{p.name:sha256_file(p) for p in folder.iterdir() if p.is_file() and p.name!='summary.json'},
        'upstream_sources_unchanged':True})
    print(table.to_string(index=False),flush=True)
    return table

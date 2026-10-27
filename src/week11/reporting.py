import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .cache import atomic,Store,digest
from .catalog import TABLES,PLOTS
from .bootstrap import paired
from .evaluation import classification_metrics
from .factorial import describe,difference_in_differences
from .oracle import surface,capture
from .plotting import render


def pareto(frame,cost='prompt_tokens',quality='task_score'):
    if frame.empty:return frame.copy()
    frame=frame.copy();flags=[]
    for _,row in frame.iterrows():
        comparable=frame[frame.dataset==row.dataset] if 'dataset' in frame else frame
        dominated=((comparable[cost]<=row[cost])&(comparable[quality]>=row[quality])&
                   ((comparable[cost]<row[cost])|(comparable[quality]>row[quality]))).any()
        flags.append(not dominated)
    frame['pareto']=flags
    return frame


def saved_rows(root):
    rows=[]
    for identity in sorted(root.glob('*/rows/identity.json')):
        if not (identity.parent.parent/'status.json').exists():continue
        obj=json.loads(identity.read_text())
        if obj.get('synthetic'):raise ValueError('Synthetic rows cannot enter real results')
        rows.extend(Store(identity.parent,obj).rows())
    return rows


def summarize(frame,retrieval=False):
    if frame.empty:return frame.copy()
    keys=[x for x in ['dataset','task_type','action_id','experiment_id','reader','reader_variant','retriever','evidence_granularity','selector',
        'budget','compressor','passage_representation','strategy','k'] if x in frame]
    numeric=[c for c in frame.select_dtypes(include=['number','bool']).columns if c not in keys and c not in ['synthetic']]
    summary=frame.groupby(keys,dropna=False)[numeric].mean().reset_index()
    counts=frame.groupby(keys,dropna=False).size().reset_index(name='n_queries')
    summary=summary.merge(counts,on=keys)
    if retrieval:
        valid=frame.assign(valid_gold_query=frame.any_gold_recall.notna()).groupby(keys,dropna=False).valid_gold_query.sum().reset_index(name='gold_metric_denominator')
        summary=summary.merge(valid,on=keys)
    if 'task_score' in summary:
        summary['quality_per_1000_tokens']=summary.task_score/summary.actual_evidence_tokens.replace(0,np.nan)*1000
        summary['requested_budget_numeric']=pd.to_numeric(summary.budget,errors='coerce')
        if 'classification_accuracy' in frame:
            for index,row in summary[summary.dataset=='SciFact'].iterrows():
                data=frame[(frame.dataset==row.dataset)&(frame.action_id==row.action_id)]
                summary.loc[index,'macro_f1']=classification_metrics(data.gold.tolist(),data.prediction.str.strip().str.upper().tolist())['macro_f1']
                summary.loc[index,'invalid_label_rate']=(~data.prediction.str.strip().str.upper().isin(['SUPPORT','REFUTE','NOT_ENOUGH_INFO'])).mean()
    return summary


def comparisons(frame,cfg,retrieval=False):
    if frame.empty:return pd.DataFrame()
    records=[]
    for dataset,ds in frame.groupby('dataset'):
        if retrieval:
            ds=ds[ds.k==6].copy();ds['task_score']=ds.evidence_coverage
            ds=ds.dropna(subset=['task_score'])
        groups={key:g for key,g in ds.groupby('action_id')};done=set()
        for action,g in groups.items():
            row=g.iloc[0];controls=[]
            for other,h in groups.items():
                r=h.iloc[0]
                if action==other:continue
                if retrieval:
                    eligible=r.retriever=='BM25' and r.evidence_granularity==row.evidence_granularity
                else:
                    same_variant=r.get('reader_variant','PUBLIC_TEXT_READER')==row.get('reader_variant','PUBLIC_TEXT_READER')
                    eligible=(same_variant and r.reader==row.reader and r.evidence_granularity=='NONE') or (
                        r.reader==row.reader and r.evidence_granularity=='PASSAGE' and r.selector=='semantic' and r.budget==row.budget and r.compressor=='C0'
                        and same_variant and r.retriever==row.retriever and r.get('retrieval_width')==row.get('retrieval_width')) or (
                        same_variant and r.reader=='Mistral' and row.reader!='Mistral' and all(r[x]==row[x] for x in ['evidence_granularity','selector','budget','compressor','retriever','retrieval_width','strategy']))
                if eligible:controls.append((other,h))
            for other,h in controls:
                key=tuple(sorted([action,other]))
                if key in done:continue
                result=paired(g.to_dict('records'),h.to_dict('records'),n=cfg['bootstrap_resamples'],seed=cfg['seed'])
                records.append(dict(dataset=dataset,action_a=action,action_b=other,
                    metric='evidence_coverage' if retrieval else 'classification_accuracy' if dataset=='SciFact' else 'answer_f1',
                    family='secondary' if not retrieval and h.iloc[0].evidence_granularity=='NONE' else 'primary',**result));done.add(key)
    return pd.DataFrame(records)


def oracle_frames(frame,cfg):
    summaries=[];per=[];distributions=[];concentrations=[];curves=[];status=[];captures=[]
    if frame.empty:return {}
    # Include a global within-dataset action oracle and separate within-reader surfaces.
    for dataset,ds in frame.groupby('dataset'):
        ds=ds.copy();ds['reader_system']=ds.reader.where(ds.compressor!='C3',ds.reader+'/SARA-specialist')
        scopes=[('ALL_READERS',ds)]+list(ds.groupby('reader_system'))
        for reader,g in scopes:
            try:obj=surface(g.to_dict('records'),cfg['epsilon'])
            except ValueError as exc:
                status.append(dict(dataset=dataset,reader=reader,status='INCOMPLETE_PAIRED_SURFACE',reason=str(exc)));continue
            summaries.append(dict(dataset=dataset,reader=reader,**{k:v for k,v in obj.items() if k in
                ['best_fixed','best_fixed_score','exact_oracle','epsilon_oracle','oracle_headroom','epsilon']}))
            per.extend(dict(r,oracle_scope=reader) for r in obj['per_query'])
            distributions.extend(dict(dataset=dataset,reader=reader,action=k,count=v) for k,v in obj['epsilon_distribution'].items())
            concentrations.extend(dict(dataset=dataset,reader=reader,**r) for r in obj['concentration'])
            curves.extend(dict(dataset=dataset,reader=reader,**r) for r in obj['curve'])
            if 'strategy' in g:
                fixed=g[g.strategy.isin(['FIXED','H0'])]
                if not fixed.empty:
                    try:fixed_oracle=surface(fixed.to_dict('records'),cfg['epsilon'])
                    except ValueError:fixed_oracle=None
                    if fixed_oracle:
                        membership=set(fixed.query_id)
                        for candidate,c in g[~g.strategy.isin(['FIXED','H0'])].groupby('action_id'):
                            if set(c.query_id)!=membership:continue
                            captures.append(dict(dataset=dataset,reader=reader,candidate=candidate,
                                candidate_score=c.task_score.mean(),best_fixed=fixed_oracle['best_fixed_score'],
                                fixed_action_oracle=fixed_oracle['exact_oracle'],
                                OCR=capture(c.task_score.mean(),fixed_oracle['best_fixed_score'],fixed_oracle['exact_oracle']),
                                scope='Retrospective fixed-action ceiling; candidate is not a learned controller'))
    return dict(oracle_summary_df=pd.DataFrame(summaries),per_query_oracle_df=pd.DataFrame(per),
        oracle_action_distribution_df=pd.DataFrame(distributions),oracle_gain_concentration_df=pd.DataFrame(concentrations),
        oracle_gain_curve_df=pd.DataFrame(curves),oracle_status_df=pd.DataFrame(status),oracle_capture_df=pd.DataFrame(captures))


def report(phase,rows,cfg,directory,setup=None,synthetic=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows)
    if not frame.empty and not synthetic and frame.get('synthetic',pd.Series(False,index=frame.index)).any():
        raise ValueError('Fabricated outcomes cannot be included in real tables')
    frames={name:pd.DataFrame() for name in TABLES[phase]}
    condition_path=directory/'condition_status_df.csv'
    if condition_path.exists():frames['condition_status_df']=pd.read_csv(condition_path)
    if phase=='0':frames.update(setup or {})
    elif phase in ('A','B'):
        summary=summarize(frame,True);pair=comparisons(frame,cfg,True)
        for name in TABLES[phase]:frames[name]=frame.copy() if name.startswith('per_query') or name=='recall_at_k_df' else pair if 'pairwise' in name else summary.copy()
        if not frame.empty:frames['first_rank_df']=frame[frame.k==10].copy()
        if phase=='B' and not frame.empty:
            overlaps=[]
            for (dataset,qid),g in frame[frame.k==10].groupby(['dataset','query_id']):
                for (_,a),(_,b) in itertools.product(g.iterrows(),repeat=2):
                    aa=set(a.top_ids);bb=set(b.top_ids)
                    overlaps.append(dict(dataset=dataset,query_id=qid,representation_a=a.passage_representation,
                        representation_b=b.passage_representation,overlap=len(aa&bb)/max(1,len(aa|bb))))
            frames['topk_overlap_df']=frames['ranking_agreement_df']=pd.DataFrame(overlaps)
            direct=frame[(frame.passage_representation=='P0')&(frame.k==6)][['dataset','query_id','evidence_coverage']].rename(columns={'evidence_coverage':'direct_coverage'})
            delta=frame[frame.k==6].merge(direct,on=['dataset','query_id'])
            delta['delta_vs_direct']=delta.evidence_coverage-delta.direct_coverage
            frames['per_query_representation_df']=delta
            frames['weighted_improvement_df']=delta[delta.passage_representation=='P3'].copy()
            frames['recall_comparison_df']=summary[summary.k==6].copy()
    else:
        block_frame=frame.copy()
        if not frame.empty:
            frame=frame.drop_duplicates(['dataset','query_id','action_id'])
        summary=summarize(frame);pair=comparisons(frame,cfg);effects,interactions=describe(frame)
        if not summary.empty:
            frames['specialist_baseline_df']=summary[summary.compressor=='C3'].copy()
            frames['portable_reader_summary_df']=summary[summary.compressor!='C3'].copy()
        if not frame.empty and 'answer_type' in frame:
            frames['answer_type_breakdown_df']=frame.groupby(['dataset','reader','action_id','answer_type'],dropna=False).agg(
                task_score=('task_score','mean'),n_queries=('query_id','size')).reset_index()
        front=pareto(summary[summary.compressor!='C3']) if not summary.empty else summary
        for name in TABLES[phase]:
            frames[name]=frame.copy() if 'per_query' in name else pair.copy() if any(x in name for x in ['pairwise','statistical_comparisons']) else interactions.copy() if 'interaction' in name else effects.copy() if name=='factorial_effects_df' else front.copy() if 'pareto' in name else summary.copy()
        frames.update(oracle_frames(frame,cfg))
        if phase=='G':
            allocations=[]
            for row in rows:
                for kind,tokens in row.get('allocation_tokens',{}).items():allocations.append(dict(dataset=row['dataset'],strategy=row['strategy'],unit_type=kind,tokens=tokens))
            frames['hierarchy_budget_allocation_df']=pd.DataFrame(allocations)
            frames['hierarchy_oracle_df']=frames.get('oracle_summary_df',pd.DataFrame())
            from .protocol import ROOT,code_identity
            candidates=[ROOT/cfg['output_root']/'granularity_budget/c_summary_df.csv'] if not synthetic else sorted(
                (ROOT/cfg['output_root']/'smoke/C').glob('*/c_summary_df.csv'),key=lambda p:p.stat().st_mtime,reverse=True)
            for path in candidates:
                if not path.exists():continue
                if synthetic:
                    provenance=json.loads((path.parent/'manifest.json').read_text())
                    if provenance['identity']['code_hash']!=digest(code_identity()):continue
                fixed=pd.read_csv(path);fixed=fixed[(fixed.selector=='semantic')&(fixed.compressor=='C0')]
                if fixed.empty:continue
                fixed['budget']=fixed.budget.astype(str)
                best=fixed.loc[fixed.groupby(['dataset','budget']).task_score.idxmax()].assign(strategy='BEST_FIXED_RETROSPECTIVE')
                current=summary.copy();current['budget']=current.budget.astype(str)
                frames['hierarchy_fixed_comparison_df']=pd.concat([current,best],ignore_index=True);break
        if phase=='H':
            frames['interaction_summary_df']=interactions;frames['factorial_effects_df']=effects
            contrasts=[]
            for spec in cfg['interaction_contrasts']:
                groups=[frame[(frame.dataset==spec['dataset'])&(frame.action_id==action)].to_dict('records') for action in spec['actions']]
                if all(groups):contrasts.append(dict(name=spec['name'],dataset=spec['dataset'],family='exploratory',**difference_in_differences(groups,n=cfg['bootstrap_resamples'])))
            frames['paired_interaction_contrasts_df']=pd.DataFrame(contrasts)
            # The W10 fixture remains read-only and is explicitly a separate protocol.
            w10=Path('results/week10/root_cause/v1/dataset_diagnostics.json')
            if w10.exists() and not synthetic:frames['w10_stability_hook_df']=pd.DataFrame(json.loads(w10.read_text())['stability']).assign(protocol='W10_MISTRAL_SAMPLING_ONLY')
        if phase=='I':
            for axis in ['dataset','reader','budget','evidence_granularity']:
                name='best_by_'+('granularity' if axis=='evidence_granularity' else axis)+'_df'
                if summary.empty:continue
                keys=['dataset'] if axis=='dataset' else ['dataset','reader','reader_variant'] if axis=='reader' else ['dataset',axis]
                frames[name]=summary.loc[summary.groupby(keys,dropna=False).task_score.idxmax()].assign(selection_status='RETROSPECTIVE_NOT_FROZEN')
            frames['retrieval_saved_df']=pd.DataFrame(cfg.get('_retrieval_saved_rows',[]))
            frames['project_trajectory_df']=pd.DataFrame(cfg['history_trajectory_records'],columns=['week','protocol','metric','value'])
            public=block_frame[block_frame.compressor!='C3'] if not block_frame.empty else block_frame
            if not public.empty:
                frames['main_granularity_summary_df']=summarize(public[public.experiment_id=='C'])
                frames['matched_reader_summary_df']=summarize(public[public.experiment_id.isin(['C','D']) & (public.compressor=='C0')])
                frames['matched_compressor_summary_df']=summarize(public[public.experiment_id=='E'])
                frames['finalist_retriever_summary_df']=summarize(public[public.experiment_id=='F'])
                fixed=frames['main_granularity_summary_df']
                if not fixed.empty:
                    fixed=fixed[(fixed.selector=='semantic') & (fixed.compressor=='C0')]
                    if not fixed.empty:
                        best=fixed.loc[fixed.groupby(['dataset','budget'],dropna=False).task_score.idxmax()].assign(strategy='BEST_FIXED_RETROSPECTIVE')
                        frames['hierarchy_fixed_comparison_df']=pd.concat([summarize(public[public.experiment_id=='G']),best],ignore_index=True)
        if not frame.empty:
            action_means=frame.groupby(['dataset','action_id']).task_score.mean().reset_index()
            action_means['within_dataset_rank']=action_means.groupby('dataset').task_score.rank(ascending=False)
            frames['cross_dataset_action_ranks_df']=action_means
    figures=[]
    for name,data in frames.items():data.to_csv(directory/(name+'.csv'),index=False)
    for spec in PLOTS[phase]:figures.extend(render(frames.get(spec[1],pd.DataFrame()),spec,directory/'figures',synthetic))
    atomic(directory/'figures.json',figures)
    lines=[f'# Week 11 phase {phase}', '', '**SYNTHETIC SMOKE — NO SCIENTIFIC FINDINGS**' if synthetic else 'Saved-result characterization only; no winner or deployable oracle is declared.', '',
        'Scores and statistical comparisons are dataset-specific. No mixed raw QA/verification overall F1 is computed.', '',
        '## Tables','']+[f'- `{name}.csv`: {len(data)} rows' for name,data in frames.items()]
    lines+=['','## Figures','']+[f'- {f["name"]}: `{Path(f["png"]).name}` (PNG, PDF, and source CSV)' for f in figures]
    (directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return frames,figures

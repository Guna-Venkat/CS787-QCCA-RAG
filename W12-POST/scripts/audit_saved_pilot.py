"""Descriptive audit of saved VAL pilot JSON only. No model/pipeline imports."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/post_w12/bea_qbgf/v3'
OUT=BASE/'pilot/aspect_mechanism_audit'
METHODS=['FIXED_SENTENCE','QBGF_FULL','BEA_QBGF']

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def envelopes(folder,identity,inputs):
    out={}
    for path in sorted(folder.glob('*.json')):
        value=json.loads(path.read_text());key=tuple(value['key'])
        if value['identity_hash']!=identity or value['checksum']!=digest(value['row']):raise ValueError('Invalid saved envelope: '+str(path))
        if key in out:raise ValueError('Duplicate saved key')
        out[key]=value['row'];inputs[str(path.relative_to(ROOT))]=sha(path)
    return out

def primary(rows,dataset):
    if dataset!='SciFact':return float(np.mean([r['f1'] for r in rows]))
    result=[]
    for label in ['SUPPORT','REFUTE','NOT_ENOUGH_INFO']:
        tp=sum(r['gold']==label and r['prediction']==label for r in rows)
        fp=sum(r['gold']!=label and r['prediction']==label for r in rows)
        fn=sum(r['gold']==label and r['prediction']!=label for r in rows)
        result.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return float(np.mean(result))

def outcome(delta):return 'win' if delta>1e-12 else 'loss' if delta< -1e-12 else 'tie'

def relation_summary(data,variable):
    rows=[]
    for (d,b,c,group),xs in data.groupby(['dataset','budget','comparator',variable],dropna=False,sort=True):
        a=[r['bea'] for r in xs.to_dict('records')];other=[r['baseline'] for r in xs.to_dict('records')]
        rows.append(dict(dataset=d,budget=b,comparator=c,variable=variable,stratum=group,n=len(xs),
                         wins=sum(xs.outcome=='win'),losses=sum(xs.outcome=='loss'),ties=sum(xs.outcome=='tie'),
                         primary_metric='macro_f1' if d=='SciFact' else 'answer_f1',
                         bea_primary=primary(a,d),baseline_primary=primary(other,d),primary_delta=primary(a,d)-primary(other,d),
                         mean_query_delta=float(xs.query_delta.mean()),mean_unused_fraction=float(xs.unused_fraction.mean()),
                         mean_aspect_count=float(xs.aspect_count.mean())))
    return pd.DataFrame(rows)

def main():
    assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='gagan'
    inputs={}
    freeze=json.loads((BASE/'BEA_QBGF_V1_FREEZE.json').read_text());ident=freeze['identity']
    assert digest(ident)==freeze['identity_hash']
    for name in ['BEA_QBGF_V1_FREEZE.json','PILOT_COHORT.json','pilot/summary.csv','pilot/analysis_complete.json']:
        inputs[str((BASE/name).relative_to(ROOT))]=sha(BASE/name)
    cohort=json.loads((BASE/'PILOT_COHORT.json').read_text())['selected']
    expected={(d,r['query_id']) for d,xs in cohort.items() for r in xs}
    assert len(expected)==150 and all(len(xs)==50 for xs in cohort.values())
    problems=envelopes(BASE/'cache/problems',digest(freeze['input_identity']),inputs)
    results=envelopes(BASE/'cache/results',freeze['identity_hash'],inputs)
    selections=envelopes(BASE/'cache/selections',freeze['identity_hash'],inputs)
    expected_rows={(d,q,m,b) for d,q in expected for m in METHODS for b in [256,1024]}
    assert set(results)==expected_rows and expected<=set(problems)
    assert expected_rows<=set(selections)
    # Independently reproduce every published summary cell from the saved answers.
    for r in pd.read_csv(BASE/'pilot/summary.csv').to_dict('records'):
        xs=[v for (d,q,m,b),v in results.items() if (d,b,m)==(r['dataset'],r['budget'],r['method'])]
        assert len(xs)==50 and abs(primary(xs,r['dataset'])-r['quality'])<1e-12
    aspect_rows=[];examples=[];pairs=[];mechanisms=[];full=[]
    for d,q in sorted(expected):
        p=problems[d,q];a=p['aspects'];n=len(a)
        assert n>=1 and len(p['scores']['aspect_scores'])==n
        aspect_rows.append(dict(dataset=d,query_id=q,aspect_count=n,aspects_json=json.dumps(a,ensure_ascii=False)))
        for b in [256,1024]:
            r=results[d,q,'BEA_QBGF',b];sel=selections[d,q,'BEA_QBGF',b]['selection'];bh=r['behavior']
            assert bh['aspect_count']==n and 0<=bh['represented_aspects']<=n
            assert r['source_hash']==p['source_hash']==selections[d,q,'BEA_QBGF',b]['source_hash']
            cov=[]
            for step in sel['trace']:
                if step['kind']=='SENTENCE':cov.append(step['coverage_gain'])
                else:
                    match=[x for x in sel['full_attempts'] if x['status']=='CHOSEN' and x['parent']==step['parent'] and x['step']==step['step']-1]
                    assert len(match)==1
                    cov.append(sum(x['coverage_gain'] for x in match[0]['hypothetical'] if x['accepted']))
            mechanism=dict(dataset=d,query_id=q,budget=b,aspect_count=n,represented_aspects=bh['represented_aspects'],
                           coverage_actions=sum(x>1e-12 for x in cov),coverage_actions_after_first=sum(x>1e-12 for x in cov[1:]),
                           coverage_reward=ident['parameters']['eta_cov']*sum(cov),
                           coverage_reward_after_first=ident['parameters']['eta_cov']*sum(cov[1:]),
                           selected_action_count=len(cov),full_active=bh['fidelity_upgrades']>0,stop_reason=r['stop_reason'],
                           unused_tokens=r['unused_budget'],unused_fraction=r['unused_budget']/b)
            mechanisms.append(mechanism)
            for comparator in ['FIXED_SENTENCE','QBGF_FULL']:
                old=results[d,q,comparator,b]
                v=float(r['correct']-old['correct']) if d=='SciFact' else r['f1']-old['f1']
                red_old=old['behavior']['mean_pairwise_redundancy'];red_new=bh['mean_pairwise_redundancy']
                reduction=red_old-red_new if red_old is not None and red_new is not None else np.nan
                pairs.append(dict(mechanism,comparator=comparator,query_delta=v,outcome=outcome(v),
                                  epsilon_outcome=outcome(v) if d=='SciFact' else ('help' if v>.01 else 'hurt' if v<-.01 else 'neutral'),
                                  bea=r,baseline=old,redundancy_reduction=reduction,
                                  aspect_group='1' if n==1 else '2' if n==2 else '3+',
                                  unused_group='0–25%' if mechanism['unused_fraction']<=.25 else '25–50%' if mechanism['unused_fraction']<=.5 else '50–75%' if mechanism['unused_fraction']<=.75 else '75–100%',
                                  redundancy_group='undefined' if pd.isna(reduction) else 'reduced' if reduction>1e-12 else 'increased' if reduction< -1e-12 else 'unchanged',
                                  full_group='FULL active' if mechanism['full_active'] else 'no FULL'))
    aspects=pd.DataFrame(aspect_rows);df=pd.DataFrame(pairs);mech=pd.DataFrame(mechanisms)
    distribution=aspects.groupby(['dataset','aspect_count']).size().rename('queries').reset_index();distribution['fraction']=distribution.queries/50
    overview=[]
    for (d,b),xs in mech.groupby(['dataset','budget']):
        overview.append(dict(dataset=d,budget=b,queries=len(xs),mean_extracted_aspects=xs.aspect_count.mean(),
                             one_aspect_queries=sum(xs.aspect_count==1),mean_represented_aspects=xs.represented_aspects.mean(),
                             full_queries=sum(xs.full_active),threshold_stops=sum(xs.stop_reason=='LOW_MARGINAL_VALUE'),
                             mean_unused_tokens=xs.unused_tokens.mean(),mean_unused_fraction=xs.unused_fraction.mean(),
                             mean_coverage_actions_after_first=xs.coverage_actions_after_first.mean(),
                             no_later_coverage_queries=sum(xs.coverage_actions_after_first==0),
                             mean_coverage_reward=xs.coverage_reward.mean(),mean_later_coverage_reward=xs.coverage_reward_after_first.mean()))
    overview=pd.DataFrame(overview)
    for d,xs in aspects.groupby('dataset'):
        for row in xs.sort_values('query_id').head(10).to_dict('records'):
            examples.append(dict(dataset=d,query_id=row['query_id'],count=row['aspect_count'],aspects=' | '.join(json.loads(row['aspects_json']))))
    examples=pd.DataFrame(examples)
    relations={v:relation_summary(df,v) for v in ['aspect_group','unused_group','redundancy_group','full_group','stop_reason','outcome']}
    correlations=[]
    for (d,b,c),xs in df.groupby(['dataset','budget','comparator']):
        for x in ['aspect_count','unused_tokens','unused_fraction','redundancy_reduction','full_active','coverage_actions_after_first']:
            valid=xs[[x,'query_delta']].dropna().astype(float)
            rho=valid[x].rank().corr(valid.query_delta.rank()) if len(valid)>2 and valid[x].nunique()>1 and valid.query_delta.nunique()>1 else None
            correlations.append(dict(dataset=d,budget=b,comparator=c,feature=x,n=len(valid),spearman_rho=rho,
                                     note='descriptive only; constant/undefined features masked; SciFact delta is per-claim correctness'))
    correlations=pd.DataFrame(correlations)
    stop=pd.crosstab([df.dataset,df.budget,df.comparator,df.outcome],df.stop_reason).reset_index()
    # Include actual FULL rejection/competition events, not only activation frequency.
    for d in sorted(cohort):
        for b in [256,1024]:
            xs=[selections[d,q,'BEA_QBGF',b]['selection'] for dd,q in expected if dd==d]
            attempts=[a for x in xs for a in x['full_attempts']]
            full.append(dict(dataset=d,budget=b,n_queries=50,eligible_queries=sum(any(a['eligible'] for a in x['full_attempts']) for x in xs),
                             full_queries=sum(x['fidelity_upgrades']>0 for x in xs),
                             **{status.lower():sum(a['status']==status for a in attempts) for status in ['CHOSEN','REJECTED_COST','REJECTED_LOW_GAIN','REJECTED_THRESHOLD','LOST_COMPETITION','BLOCKED_ONE_SENTENCE']}))
    full=pd.DataFrame(full)
    OUT.mkdir(parents=True,exist_ok=True)
    frames=dict(aspect_distribution=distribution,aspect_inventory=aspects,aspect_examples=examples,mechanism_overview=overview,
                correlations=correlations,stop_reasons_by_outcome=stop,full_diagnostics=full,query_mechanisms=mech,
                **{'quality_by_'+k:v for k,v in relations.items()})
    public=df.drop(columns=['bea','baseline'])
    frames['paired_query_diagnostics']=public
    frames['aspect_counts_by_outcome']=df.groupby(['dataset','budget','comparator','outcome','aspect_count']).size().rename('queries').reset_index()
    frames['outcome_mechanisms']=df.groupby(['dataset','budget','comparator','outcome'],as_index=False).agg(queries=('query_id','size'),mean_aspect_count=('aspect_count','mean'),mean_unused_tokens=('unused_tokens','mean'),mean_unused_fraction=('unused_fraction','mean'),mean_redundancy_reduction=('redundancy_reduction','mean'),full_queries=('full_active','sum'))
    for name,frame in frames.items():frame.to_csv(OUT/(name+'.csv'),index=False)
    # Scores and source objects are never rewritten. Confirm every input stayed byte-identical.
    changed=[name for name,h in inputs.items() if sha(ROOT/name)!=h]
    if changed:raise ValueError('Input changed during read-only audit: '+str(changed))
    manifest=dict(source='existing pilot data only',inputs=inputs,model_generations=0,parameter_changes=False,
                  logical_answer_rows=900,paired_diagnostic_rows=len(df),cohort_queries=150,
                  script_sha256=sha(Path(__file__)),report_script_sha256=sha(Path(__file__).with_name('write_aspect_audit_report.py')),scientific_identity=freeze['identity_hash'])
    (OUT/'INPUT_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Save compact building blocks for report drafting; the report interprets these tables.
    print('Verified saved input files:',len(inputs),'unchanged')
    print('\nASPECT DISTRIBUTION\n',distribution.to_markdown(index=False))
    print('\nOVERVIEW\n',overview.to_markdown(index=False))
    print('\nQUALITY BY ASPECT COUNT\n',relations['aspect_group'].to_markdown(index=False))
    print('\nWINS/LOSSES\n',relations['outcome'].to_markdown(index=False))
    print('\nFULL\n',full.to_markdown(index=False))
    import runpy
    runpy.run_path(str(Path(__file__).with_name('write_aspect_audit_report.py')),run_name='__main__')

if __name__=='__main__':main()

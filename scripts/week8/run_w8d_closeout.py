from __future__ import annotations
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse,json,time
from dataclasses import asdict
import numpy as np,pandas as pd,torch
from src.week8.closeout import *


def main():
    parser=argparse.ArgumentParser(description='W8-D cached selector closeout; no SARA extraction')
    parser.add_argument('--device',default='cuda');parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();torch.set_num_threads(4)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    cfg=TrainingConfig(device=args.device)
    stamp=hashlib.sha256((Path(__file__).read_text()+(ROOT/'src/week8/closeout.py').read_text()+
                         json.dumps(asdict(cfg),sort_keys=True)).encode()).hexdigest()
    if (out/'run_stamp.json').exists():
        if json.loads((out/'run_stamp.json').read_text())['stamp']!=stamp:raise ValueError('Closeout code/config changed; use a fresh directory')
    (out/'run_stamp.json').write_text(json.dumps({'stamp':stamp,'training_config':asdict(cfg)},indent=2))
    fit=cached_split('fit');fit_scores={};params={}
    for name in ['S3','S5','S6']:fit_scores[name]=heuristic_score(fit,name)
    for name in ['S7','S8']:
        fit_scores[name],params[name]=heuristic_oof(fit,name)
    saved_fit=attach_saved_predictions(fit,'fit');fit_scores['U4-A']=saved_fit['utility_score']
    emb=encoder_cache(fit,'fit',cfg,out/'encoder_cache')
    learned={}
    for name,cols in [('U4-A-replay',BASE_COLUMNS),('U4-B',BASE_COLUMNS+ATTR_COLUMNS)]:
        learned[name]=hybrid_oof(fit,emb,cols,cfg,out/name)
        fit_scores[name]=learned[name]['utility_score']
        pd.concat([fit[KEYS],pd.DataFrame(learned[name])],axis=1).to_csv(out/f'{name}_fit_oof_predictions.csv',index=False)
    fit_metrics={name:selector_metrics(fit,s) for name,s in fit_scores.items()}
    fit_bootstrap={name:bootstrap_difference(fit,s,fit_scores['S3']) for name,s in fit_scores.items() if name!='S3'}
    candidates=['S5','S6','S7','S8','U4-B']
    best=max(candidates,key=lambda n:fit_metrics[n]['always_promote_f1'])
    frozen={'parameters':params,'candidate_for_confirmation':best,'selection_split':'FIT paper OOF only',
        'attribution':'per-token logprob change; empty drafts fall back to DraftSemantic for S5',
        'quality_margin':.003,'hurt_reduction_margin':.02,
        'success_rule':'quality >=.003 with paired 95% lower bound >0, OR hurt reduction >=.02 with upper bound <0 AND F1 lower bound >-.003',
        'fit_metrics':fit_metrics,'fit_paper_bootstrap_vs_S3':fit_bootstrap,
        'frozen_before_internal_val':True}
    (out/'fit_frozen_decisions.json').write_text(json.dumps(frozen,indent=2))
    print('FROZEN FIT choices:',best,params,flush=True)
    # Refit exactly one new architecture (U4-B) on full FIT. Replay is a matched
    # implementation check on OOF, not another architecture search.
    model,prep=train_cached_hybrid(fit,emb,BASE_COLUMNS+ATTR_COLUMNS,cfg)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'preprocessing':prep,'columns':BASE_COLUMNS+ATTR_COLUMNS,'training_config':asdict(cfg)},out/'selected_u4b.pt')
    val=cached_split('internal_val');ev=encoder_cache(val,'internal_val',cfg,out/'encoder_cache')
    val_scores={n:heuristic_score(val,n) for n in ['S3','S5','S6']}
    for name in ['S7','S8']:val_scores[name]=heuristic_score(val,name,**params[name]['frozen_full_fit_parameters'])
    val_saved=attach_saved_predictions(val,'internal_val');val_scores['U4-A']=val_saved['utility_score']
    val_learned=predict_cached_hybrid(model,prep,val,ev,BASE_COLUMNS+ATTR_COLUMNS,cfg)
    val_scores['U4-B']=val_learned['utility_score']
    del model
    pd.concat([val[KEYS],pd.DataFrame(val_learned)],axis=1).to_csv(out/'U4-B_internal_val_predictions.csv',index=False)
    val_metrics={name:selector_metrics(val,s) for name,s in val_scores.items()}
    val_bootstrap={name:bootstrap_difference(val,s,val_scores['S3']) for name,s in val_scores.items() if name!='S3'}
    confirm=val_bootstrap[best]
    quality=confirm['f1']['difference']>=.003 and confirm['f1']['ci95'][0]>0
    safety=(confirm['hurt']['difference']<=-.02 and confirm['hurt']['ci95'][1]<0 and confirm['f1']['ci95'][0]>-.003)
    survives=bool(quality or safety)
    # S6 can survive as a cheap MIG baseline without endorsing attribution.
    uses_attribution=best in ['S5','U4-B'] or (best in ['S7','S8'] and params[best]['frozen_full_fit_parameters']['alpha']>0)
    decision={'candidate_frozen_on_fit':best,'confirmed_quality':bool(quality),'confirmed_safety':bool(safety),
        'candidate_survives':survives,'online_attribution_survives':bool(survives and uses_attribution),
        'verdict':'positive' if survives and uses_attribution else 'no demonstrated attribution benefit in tested variants',
        'interpretation':'Dependence on compressed evidence is distinct from benefit of exposing full text.',
        'do_not_retune_on_internal_val':True}
    summary={**frozen,'internal_val_metrics':val_metrics,'internal_val_paper_bootstrap_vs_S3':val_bootstrap,'decision':decision}
    (out/'closeout_summary.json').write_text(json.dumps(summary,indent=2))
    for split,df,scores,metrics in [('fit',fit,fit_scores,fit_metrics),('internal_val',val,val_scores,val_metrics)]:
        pd.concat([df[KEYS],pd.DataFrame(scores)],axis=1).to_csv(out/f'{split}_selector_scores.csv',index=False)
        pd.json_normalize([{'selector':n,**m} for n,m in metrics.items()]).to_csv(out/f'{split}_metrics.csv',index=False)
    print(json.dumps(decision,indent=2),flush=True)
    print(pd.DataFrame(val_metrics).T[['always_promote_f1','help_rate','hurt_rate','oracle_gated_shr']].to_string(),flush=True)


if __name__=='__main__':main()

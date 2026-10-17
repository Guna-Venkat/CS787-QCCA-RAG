"""Cached W8-D selector closeout: paper OOF, frozen choices, paired paper bootstrap."""
from __future__ import annotations
import gc
import hashlib
import itertools
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from src.week8.data import PROJECT_ROOT
from src.week8.generator_features import source_fingerprint, validate_feature_rows
from src.week8.utility_dataset import get_paper_grouped_splits
from src.week8.utility_training import (
    TrainingConfig, HybridSelector, load_surface, targets_for, assert_training_split,
    load_selected_model, encode_group,
)
from src.week8.utility_models import MultitaskUtilityLoss

KEYS = ['question_id', 'paper_id', 'semantic_rank', 'candidate_passage_index']
ATTR_COLUMNS = ['masking_delta_logp', 'masking_delta_logp_per_token',
                'masking_target_tokens', 'masking_available']
BASE_COLUMNS = json.loads((PROJECT_ROOT / 'results/week8/w8c_utility/w8c_frozen_selector_config.json').read_text())['feature_columns']
OUT = PROJECT_ROOT / 'results/week8/w8d_closeout'


def cached_split(split):
    df = load_surface(split)
    folder = PROJECT_ROOT / 'results/week8/w8d_attribution/full'
    feat = pd.read_csv(folder / f'features_{split}.csv', dtype={'question_id': str, 'paper_id': str})
    fingerprint, _ = source_fingerprint(split)
    provenance = json.loads((folder / f'provenance_{split}.json').read_text())
    if provenance['source_fingerprint'] != fingerprint:
        raise ValueError('W8-D source fingerprint changed; inspect provenance before using features')
    refs = {str(q): g for q,g in df.groupby('question_id')}
    for q,g in feat.groupby('question_id'):
        validate_feature_rows(g.to_dict('records'), refs[str(q)], fingerprint)
    if set(feat.question_id) != set(df.question_id) or len(feat) != len(df):
        raise ValueError('Incomplete cached feature coverage')
    extra = [c for c in feat if c not in df or c in KEYS]
    df = df.merge(feat[extra], on=KEYS, how='left', validate='one_to_one', sort=False)
    drafts = df.draft_answer.astype(str)
    df['draft_length_words'] = drafts.map(lambda x: len(x.lower().split()))
    df['draft_empty'] = drafts.map(lambda x: int(not x.lower().split()))
    df['draft_query_overlap'] = [len(set(q.lower().split()) & set(d.lower().split())) /
        max(len(set(q.lower().split()) | set(d.lower().split())), 1) for q,d in zip(df.question,drafts)]
    if not np.isfinite(df[BASE_COLUMNS].to_numpy(float)).all():
        raise ValueError('Missing/nonfinite cached base features')
    if not np.allclose(df.a6_draft_sim,df.draft_relevance_ry,atol=1e-6):
        raise ValueError('Draft semantic feature definitions disagree')
    df['best_delta'] = df.groupby('question_id').delta_f1.transform('max')
    return df.reset_index(drop=True)


def attach_saved_predictions(df, split):
    path = PROJECT_ROOT / 'results/week8/w8c_utility' / (
        'U4-A_fit_oof_predictions.csv' if split=='fit' else 'selected_internal_val_predictions.csv')
    pred = pd.read_csv(path,dtype={'question_id':str,'paper_id':str})
    merged = df[KEYS].merge(pred,on=KEYS,how='left',validate='one_to_one',sort=False)
    cols = ['utility_score','prob_help','prob_hurt']
    if not np.isfinite(merged[cols].to_numpy()).all():
        raise ValueError('Missing saved U4-A predictions')
    return {c:merged[c].to_numpy() for c in cols}


def chosen_rows(df, scores):
    scores=np.asarray(scores,float)
    if scores.shape != (len(df),) or not np.isfinite(scores).all():
        raise ValueError('One finite score is required per candidate')
    work=df.assign(_score=scores)
    # Deterministic semantic-rank tie break, independent of input row order.
    selected=work.sort_values('semantic_rank').loc[lambda x:x.groupby('question_id')._score.idxmax()]
    return selected.sort_values('question_id').copy()


def selector_metrics(df,scores):
    s=chosen_rows(df,scores);e=s[s.best_delta>.01]
    best=s.best_delta.to_numpy();d=s.delta_f1.to_numpy();mask=best>.01
    denominator=np.where(mask,best,0).mean()
    def subset(g):
        if len(g)==0:return {'n':0}
        return dict(n=len(g),help_rate=float((g.delta_f1>.01).mean()),
            hurt_rate=float((g.delta_f1<-.01).mean()),mean_delta=float(g.delta_f1.mean()),
            mean_regret=float((g.best_delta-g.delta_f1).clip(lower=0).mean()),
            top1_tied_best=float(np.isclose(g.delta_f1,g.best_delta,atol=1e-8,rtol=0).mean()),
            near_oracle_rate=float(((g.best_delta-g.delta_f1)<=.01).mean()))
    return dict(always_promote_f1=float(s.promoted_f1.mean()),base_f1=float(s.base_f1.mean()),
        oracle_gated_shr=float(np.where(mask,d,0).mean()/denominator) if denominator>0 else 0.,
        **subset(s),escalation=subset(e),
        final_prompt_tokens=float(s.promoted_context_tokens.mean()),
        cumulative_prompt_tokens=float((s.base_context_tokens+s.promoted_context_tokens).mean()))


def bootstrap_difference(df,a,b,n=10000):
    x=chosen_rows(df,a);y=chosen_rows(df,b)
    if x.question_id.tolist()!=y.question_id.tolist():raise ValueError('Unpaired queries')
    diff=pd.DataFrame({'paper':x.paper_id.to_numpy(),
        'f1':x.promoted_f1.to_numpy()-y.promoted_f1.to_numpy(),
        'hurt':(x.delta_f1.to_numpy()<-.01).astype(float)-(y.delta_f1.to_numpy()<-.01).astype(float),
        'help':(x.delta_f1.to_numpy()>.01).astype(float)-(y.delta_f1.to_numpy()>.01).astype(float)})
    groups=diff.groupby('paper').agg(n=('f1','size'),f1=('f1','sum'),hurt=('hurt','sum'),help=('help','sum'))
    ix=np.random.default_rng(42).integers(0,len(groups),size=(n,len(groups)))
    denom=groups.n.to_numpy()[ix].sum(1)
    result={}
    for key in ['f1','hurt','help']:
        vals=groups[key].to_numpy()[ix].sum(1)/denom
        result[key]={'difference':float(diff[key].mean()),'ci95':np.quantile(vals,[.025,.975]).tolist()}
    return result


def heuristic_score(df,name,alpha=0.,beta=0.):
    semantic=df.draft_relevance_ry.to_numpy(float)
    # Per-token attribution prevents answer length from defining the scale.
    attr=df.masking_delta_logp_per_token.fillna(0).to_numpy(float)
    if name=='S3':return semantic
    if name=='S5':return np.where(df.masking_available.to_numpy(bool),attr,semantic)
    if name=='S6':return df.mig_score.to_numpy(float)
    if name=='S7':return semantic+alpha*attr
    if name=='S8':return semantic+alpha*attr+beta*df.mig_score.to_numpy(float)
    raise ValueError(name)


def tune_heuristic(df,name):
    grid=itertools.product([0.,.1,.3,1.,3.],[0.,.1,.3,1.] if name=='S8' else [0.])
    best=None
    for alpha,beta in grid:
        s=chosen_rows(df,heuristic_score(df,name,alpha,beta))
        rank=(float(s.promoted_f1.mean()),-alpha,-beta)
        if best is None or rank>best[0]:best=(rank,{'alpha':alpha,'beta':beta})
    return best[1]


def heuristic_oof(df,name):
    scores=np.full(len(df),np.nan);folds=[]
    for train,val in get_paper_grouped_splits(df,5):
        params=tune_heuristic(df.iloc[train],name)
        scores[val]=heuristic_score(df.iloc[val],name,**params)
        folds.append(params)
    return scores,{'fold_parameters':folds,'frozen_full_fit_parameters':tune_heuristic(df,name)}


def encoder_cache(df,split,cfg,directory):
    """Only label-independent frozen encoder features are shared across folds."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    texts=df[['question','draft_answer','candidate_text']].to_json(orient='records')
    stamp=hashlib.sha256((texts+json.dumps(asdict(cfg),sort_keys=True)).encode()).hexdigest()
    path=directory/f'{split}_encoder.npz'
    if path.exists():
        with np.load(path) as z:
            if str(z['fingerprint'])!=stamp:raise ValueError('Stale frozen encoder cache')
            return z['embeddings']
    _,model,_,_=load_selected_model(PROJECT_ROOT/'results/week8/w8c_utility',cfg.device)
    model.eval();values=np.zeros((len(df),model.encoder.backbone.config.hidden_size),np.float32)
    with torch.inference_mode():
        for _,g in tqdm(df.groupby('question_id'),desc=f'Frozen CE {split}'):
            g=g.sort_values('semantic_rank')
            encoded=encode_group(model,g,cfg)
            encoded={k:v.to(cfg.device) for k,v in encoded.items() if k!='numeric_features'}
            hidden=model.encoder.backbone(**encoded).last_hidden_state[:,0,:]
            values[g.index]=hidden.cpu().numpy()
    np.savez_compressed(path,embeddings=values,fingerprint=stamp)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return values


def train_cached_hybrid(df,embeddings,columns,cfg):
    assert_training_split(df)
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
    preprocessing=make_pipeline(SimpleImputer(strategy='median'),StandardScaler()).fit(df[columns])
    # Instantiate the same frozen encoder and fusion architecture as W8-C. Only
    # fusion is trained; cached CLS tensors replace repeated identical encodings.
    container=HybridSelector(cfg,columns,preprocessing)
    fusion=container.fusion.to(cfg.device);del container
    optimizer=torch.optim.AdamW(fusion.parameters(),lr=cfg.learning_rate,weight_decay=.01)
    loss_fn=MultitaskUtilityLoss()
    numeric=torch.tensor(preprocessing.transform(df[columns]),dtype=torch.float32,device=cfg.device)
    ce=torch.tensor(embeddings,dtype=torch.float32,device=cfg.device)
    work=df.reset_index(drop=True)
    groups=[g.sort_values('semantic_rank') for _,g in work.groupby('question_id')]
    targets=[targets_for(g,cfg.device) for g in groups]
    rng=np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        fusion.train();losses=[]
        for i in rng.permutation(len(groups)):
            ix=groups[i].index.to_numpy()
            optimizer.zero_grad(set_to_none=True)
            # W8-C also executes its unused encoder-head dropout before fusion.
            torch.nn.functional.dropout(torch.zeros((len(ix),64),device=cfg.device),p=.1,training=True)
            outputs=fusion(ce[ix],numeric[ix]);loss=loss_fn(outputs,targets[i])['loss_total']
            if not torch.isfinite(loss):raise ValueError('Nonfinite U4 closeout loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(fusion.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        print(f'cached hybrid epoch {epoch+1}: {np.mean(losses):.5f}',flush=True)
    return fusion.eval(),preprocessing


def predict_cached_hybrid(model,prep,df,embeddings,columns,cfg):
    numeric=torch.tensor(prep.transform(df[columns]),dtype=torch.float32,device=cfg.device)
    ce=torch.tensor(embeddings,dtype=torch.float32,device=cfg.device)
    result={k:np.zeros(len(df)) for k in ['utility_score','prob_help','prob_hurt','pred_delta_f1']}
    with torch.inference_mode():
        for start in range(0,len(df),256):
            pred=model(ce[start:start+256],numeric[start:start+256])
            for k in result:result[k][start:start+256]=pred[k].cpu().numpy()
    return result


def hybrid_oof(df,embeddings,columns,cfg,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={k:np.full(len(df),np.nan) for k in ['utility_score','prob_help','prob_hurt','pred_delta_f1']}
    for fold,(tr,va) in enumerate(get_paper_grouped_splits(df,cfg.n_splits)):
        path=output/f'fold_{fold}.npz'
        # Each fold commits scores only after all three epochs and validation.
        if path.exists():
            with np.load(path) as saved:
                if not np.array_equal(saved['validation_rows'],va):raise ValueError('Fold cache rows changed')
                pred={k:saved[k] for k in result}
        else:
            print(f'{output.name} fold {fold+1}/{cfg.n_splits}',flush=True)
            model,prep=train_cached_hybrid(df.iloc[tr],embeddings[tr],columns,cfg)
            pred=predict_cached_hybrid(model,prep,df.iloc[va],embeddings[va],columns,cfg)
            np.savez_compressed(path,validation_rows=va,**pred)
            del model;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        for k in result:result[k][va]=pred[k]
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Incomplete closeout OOF scores')
    return result

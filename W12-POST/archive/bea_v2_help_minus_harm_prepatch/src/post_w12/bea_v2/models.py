"""CPU-only group-cross-fitted logistic models; fixed architecture, no search."""
import io,warnings
import numpy as np
import pandas as pd
import joblib,sklearn
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.exceptions import ConvergenceWarning
from .protocol import *
from .data import audit,feature_rows

def fit_binary(x,y,c):
    if len(set(y))<2:raise ValueError('Degenerate one-class training target; no fabricated probabilities')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=c['model']['C'],penalty='l2',solver='lbfgs',max_iter=c['model']['max_iter'],random_state=c['seed'],class_weight=None))
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
    return model

def save_model(path,value):
    stream=io.BytesIO();joblib.dump(value,stream);data=stream.getvalue();path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError('Incompatible existing predictor; choose new output version')
    else:
        tmp=path.with_suffix('.partial');tmp.write_bytes(data);os.replace(tmp,path)

def verify_models(c):
    m=read(root(c)/'utility_model/MODELS.json')
    if m['sklearn_version']!=sklearn.__version__:raise ValueError('Predictor library version changed; use a new output version')
    if m['implementation']!=implementation():raise ValueError('V2 implementation changed after model training; use a new version')
    if m['feature_hash']!=digest(FEATURES) or m['model_config']!=c['model']:raise ValueError('Model/schema configuration mismatch')
    if m['fold_hash']!=sha(root(c)/'CROSSFIT_FOLDS.json'):raise ValueError('Fold identity mismatch')
    for name,h in dict(m['files'],**m.get('diagnostic_files',{})).items():
        if sha(root(c)/name)!=h:raise ValueError('Predictor checksum mismatch')
    return m

@exclusive
def train(c=None):
    c=c or load();frame,cohort=audit.__wrapped__(c)
    path=root(c)/'utility_model/MODELS.json'
    if path.exists():return verify_models(c)
    assignments={};folds={}
    # Membership is determined without outcome labels, on the frozen full cohort.
    for d in DATASETS:
        members=sorted(cohort['selected'][d],key=lambda r:r['query_id']);groups=[r['group'] for r in members]
        if len(set(groups))<c['folds']:raise ValueError('Insufficient unique groups')
        assign={}
        for fold,(tr,te) in enumerate(GroupKFold(n_splits=c['folds']).split(np.zeros(len(groups)),groups=groups)):
            if set(groups[i] for i in tr)&set(groups[i] for i in te):raise ValueError('Group leakage')
            for i in te:assign[members[i]['query_id']]=fold
        assignments[d]=assign;folds[d]={str(i):[r for r in members if assign[r['query_id']]==i] for i in range(c['folds'])}
    obj=dict(seed=c['seed'],algorithm='5-fold GroupKFold, deterministic sorted query input, no outcome stratification',assignments=assignments,members=folds)
    write(root(c)/'CROSSFIT_FOLDS.json',obj);write(root(c)/'utility_model/fold_assignments.json',obj)
    predictions=[];metrics=[];bins=[];files={}
    for d in DATASETS:
        data=frame[frame.dataset==d].copy();data['fold']=data.query_id.map(assignments[d])
        for fold in range(c['folds']):
            tr=data[data.fold!=fold];te=data[data.fold==fold]
            hold={r['group'] for r in folds[d][str(fold)]}
            if set(tr.group)&hold or set(tr.query_id)&set(te.query_id):raise ValueError('Training saw held-out group/query')
            if te.empty:raise ValueError('No held-out labels for diagnostics')
            models={};x=tr[FEATURES].to_numpy(float);xt=te[FEATURES].to_numpy(float)
            for target in ['HELPFUL','HARMFUL']:models[target]=fit_binary(x,(tr.category==target).astype(int).to_numpy(),c)
            bundle=dict(models=models,dataset=d,fold=fold,feature_names=FEATURES,training_groups=sorted(set(tr.group)),heldout_groups=sorted(hold),training_queries=sorted(set(tr.query_id)),feature_hash=digest(FEATURES))
            file=root(c)/f'utility_model/fold_{d}_{fold}.joblib';save_model(file,bundle);files[str(file.relative_to(root(c)))]=sha(file)
            ph=models['HELPFUL'].predict_proba(xt)[:,1];pm=models['HARMFUL'].predict_proba(xt)[:,1]
            for j,r in enumerate(te.to_dict('records')):
                predictions.append({k:r[k] for k in ['dataset','query_id','group','method','budget','unit_id','category']}|dict(fold=fold,p_help=float(ph[j]),p_harm=float(pm[j])))
            for target,pr in [('HELPFUL',ph),('HARMFUL',pm)]:
                y=(te.category==target).astype(int).to_numpy();variable=len(set(y))==2
                metrics.append(dict(dataset=d,fold=fold,target=target,n_units=len(y),n_groups=te.group.nunique(),prevalence=float(y.mean()),
                                    roc_auc=float(roc_auc_score(y,pr)) if variable else None,
                                    pr_auc_average_precision=float(average_precision_score(y,pr)) if y.sum() else None,
                                    brier=float(brier_score_loss(y,pr)),constant_prevalence_brier=float(y.mean()*(1-y.mean()))))
                for i in range(10):
                    use=(pr>=i/10)&((pr<(i+1)/10) if i<9 else (pr<=1))
                    bins.append(dict(dataset=d,fold=fold,target=target,bin=i,count=int(use.sum()),mean_probability=float(pr[use].mean()) if use.any() else None,observed_rate=float(y[use].mean()) if use.any() else None))
            print(f'CPU cross-fit {d} fold {fold+1}/5; held-out units={len(te)}; held-out groups absent from training',flush=True)
    P=pd.DataFrame(predictions);M=pd.DataFrame(metrics);B=pd.DataFrame(bins)
    for name,df in [('crossfit_predictions',P),('fold_metrics',M),('calibration',B)]:text(root(c)/'utility_model'/(name+'.csv'),df.to_csv(index=False))
    fold_means=P.groupby(['dataset','fold','category']).agg(units=('unit_id','size'),mean_p_help=('p_help','mean'),mean_p_harm=('p_harm','mean')).reset_index()
    fold_means['prevalence']=fold_means.units/fold_means.groupby(['dataset','fold']).units.transform('sum')
    text(root(c)/'utility_model/per_fold_probability_by_actual_class.csv',fold_means.to_csv(index=False))
    means=P.groupby(['dataset','category'])[['p_help','p_harm']].mean().reset_index()
    text(root(c)/'utility_model/probability_by_actual_class.csv',means.to_csv(index=False))
    manifest=dict(label=LABEL,feature_hash=digest(FEATURES),fold_hash=sha(root(c)/'CROSSFIT_FOLDS.json'),model_config=c['model'],
                  implementation=implementation(),sklearn_version=sklearn.__version__,files=files,training_sources=read(root(c)/'TRAINING_SOURCES.json'),
                  diagnostic_files={str((root(c)/'utility_model'/(name+'.csv')).relative_to(root(c))):sha(root(c)/'utility_model'/(name+'.csv')) for name in ['crossfit_predictions','fold_metrics','calibration','probability_by_actual_class','per_fold_probability_by_actual_class']},
                  train_rows=len(frame),heldout_rows=len(P),target='historical exact sentence LOO signs; intervention-specific, not causal probabilities')
    write(path,manifest)
    text(root(c)/'utility_model/UTILITY_MODEL_REPORT.md','# Out-of-fold usefulness models\n\nCPU logistic regression, C=1, StandardScaler fit on training folds only. Separate models per dataset and binary target; no class reweighting or hyperparameter search.\n\n'+M.to_markdown(index=False)+'\n\n## Predicted probabilities by actual historical class\n\n'+means.to_markdown(index=False)+'\n\n## Per-fold actual-class probabilities\n\n'+fold_means.to_markdown(index=False)+'\n\nPR-AUC is average precision; ROC-AUC is undefined for one-class held-out targets. Calibration bins are in calibration.csv. Historical selected-unit data are context-specific and imbalanced; probabilities are predictive surrogates, not causal necessity. No gold/answer/LOO value enters the feature vector. The cohort is developmental; even cross-fitting does not make it fresh validation.\n')
    return manifest

def predict(c,d,q,group,features):
    manifest=verify_models(c);folds=read(root(c)/'CROSSFIT_FOLDS.json');fold=folds['assignments'][d].get(q)
    if fold is None:raise ValueError('Unseen query requires separately frozen all-development predictor; crossfit model cannot be substituted')
    relative=f'utility_model/fold_{d}_{fold}.joblib';bundle=joblib.load(root(c)/relative)
    if group in bundle['training_groups'] or q in bundle['training_queries'] or group not in bundle['heldout_groups']:
        raise ValueError('Held-out query/group leaked into its predictor')
    if bundle['feature_names']!=FEATURES:raise ValueError('Feature order changed')
    x=np.array([[r[k] for k in FEATURES] for r in features],float)
    if not len(x):return [],[],dict(fold=fold,model_hash=manifest['files'][relative])
    ph=bundle['models']['HELPFUL'].predict_proba(x)[:,1];pm=bundle['models']['HARMFUL'].predict_proba(x)[:,1]
    if not (np.isfinite(ph).all() and np.isfinite(pm).all() and np.all((ph>=0)&(ph<=1)) and np.all((pm>=0)&(pm<=1))):raise ValueError('Invalid probabilities')
    return ph.tolist(),pm.tolist(),dict(fold=fold,model_hash=manifest['files'][relative],feature_hash=digest(FEATURES))

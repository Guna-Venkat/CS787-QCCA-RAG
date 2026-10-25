"""Explicit audit-gated final packaging/refit; never overwrites controller OOF."""
import argparse
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
import joblib
from importlib.metadata import version
from ..provenance import ROOT,branch_guard,digest,file_hash
from ..caching import atomic_json
from ..action_verification_g import protocol,controller
from .audit import VALID_STATUSES,load_case,check,audit_models


def validate_audit(path):
    status=json.loads(Path(path).read_text())
    check(status['status'] in VALID_STATUSES and status['safe_for_final_fit'] and not status['oof_rerun_required'],
        'Final fit requires a valid matching audit; INVALID audits cannot deploy')
    check(status['no_new_features_or_models'] and status['deployment_threshold']==.70,'Frozen audit schema/threshold differs')
    protocol.verify_hashes(status['input_hashes']);protocol.verify_hashes(status['artifact_hashes'])
    protocol.verify_hashes(status['audit_code_hashes'])
    return status


def validate_destination(root,destination):
    expected=(root/'final_fit').resolve();actual=Path(destination).resolve()
    check(actual==expected,'Final deployment output must be the separate final_fit directory; OOF cannot be overwritten')
    check(not actual.exists(),'Final deployment fit already exists; preserve it and use an explicit new protocol if needed')
    return actual


def frozen_schema(columns,status,training_protocol):
    check(list(columns)==status['frozen_feature_columns']==training_protocol['features'] and len(columns)==32,
        'Final fit requires exactly the original frozen 32-feature order')
    return dict(feature_columns=list(columns),feature_count=32,feature_schema_hash=digest(list(columns)),
        no_new_features=True,metadata_excluded=['question_id','paper_id'],
        imputation='Frozen training-fold pipelines; final preprocessing fits only all FIT, never held-out data')


def export_final(audit_path,reuse=False,refit=False):
    branch_guard()
    check(bool(reuse)!=bool(refit),'Explicitly select reuse-audited-model or refit; no default training')
    status=validate_audit(audit_path);case=load_case()
    check(status['source_controller_protocol_hash']==digest(case['p']),'OOF source protocol differs from audited version')
    destination=validate_destination(case['root'],case['root']/'final_fit')
    schema=frozen_schema(case['x'].columns,status,case['p'])
    audit_models(case)  # Read-only hyperparameter/preprocessor consistency checks.
    x=case['x'];meta=case['meta']
    y=np.array([case['bundle']['groups']['F0'][q]['f1']-case['bundle']['groups']['D0'][q]['f1']>.01 for q in meta.question_id],int)
    if reuse:
        pipeline=joblib.load(case['oof']/'logistic.joblib')
    else:
        # One final FIT fit only. No OOF retraining, model search or performance evaluation.
        pipeline=controller.estimator('logistic',42)
        pipeline.fit(x,y)
    check(list(pipeline.feature_names_in_)==schema['feature_columns'],'Final fitted schema differs')
    check(float(pipeline.steps[1][1].n_samples_seen_)==1746,'Final pipeline must fit all FIT samples')
    learner=pipeline.steps[-1][1]
    check(learner.C==1 and learner.class_weight=='balanced' and learner.random_state==42,'Frozen final hyperparameters differ')
    # Save in a temporary sibling and publish only complete packaging. Never modify OOF.
    temp=Path(tempfile.mkdtemp(prefix='.final_fit_',dir=case['root']))
    try:
        if reuse:shutil.copyfile(case['oof']/'logistic.joblib',temp/'logistic_pipeline.joblib')
        else:joblib.dump(pipeline,temp/'logistic_pipeline.joblib')
        joblib.dump(learner,temp/'logistic_model.joblib')
        joblib.dump(pipeline[:-1],temp/'fitted_preprocessing.joblib')
        atomic_json(temp/'feature_schema.json',schema)
        atomic_json(temp/'threshold.json',dict(threshold=.70,source='Already selected pooled outer-OOF deployment threshold',
            source_artifact_hash=file_hash(case['oof']/'logistic_deployment.json'),
            post_selection=True,not_used_for_honest_nested_oof=True))
        queries=meta.question_id.tolist();papers=sorted(meta.paper_id.unique().tolist())
        atomic_json(temp/'training_query_ids.json',queries);atomic_json(temp/'training_paper_ids.json',papers)
        manifest=dict(schema='W9-G-final-logistic-fit-v1',complete=True,split='fit',n_queries=1746,n_papers=656,
            packaging_mode='reuse existing audited full-FIT pipeline' if reuse else 'explicit final-only refit on all FIT',
            new_estimator_fits=0 if reuse else 1,
            training_query_ids_hash=digest(queries),training_paper_ids_hash=digest(papers),
            labels_hash=digest(y.tolist()),positive_labels=int(y.sum()),negative_labels=int((1-y).sum()),
            feature_schema_hash=schema['feature_schema_hash'],threshold=.70,
            sklearn=version('scikit-learn'),config=case['cfg'],config_hash=digest(case['cfg']),
            source_controller_protocol_hash=digest(case['p']),source_oof_model_hash=file_hash(case['oof']/'logistic.joblib'),
            audit_status=status['status'],audit_artifact_hash=file_hash(audit_path),
            code_hashes={str(p.relative_to(ROOT)):file_hash(p) for p in Path(__file__).parent.glob('*.py')},
            frozen_upstream_hashes=case['bundle']['input_hashes'],
            honest_performance_source='Existing nested-paper OOF tables only; no final-model FIT predictions evaluated.',
            training_performance_reported=False,heldout_evaluation_performed=False,test_touched=False,
            artifact_hashes={p.name:file_hash(p) for p in temp.iterdir() if p.is_file()})
        atomic_json(temp/'training_manifest.json',manifest)
        protocol.verify_hashes(case['protected'])
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    print(f'Final logistic package: {destination}. No training-set/held-out performance was evaluated.')
    return destination


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--reuse-audited-model',action='store_true')
    group.add_argument('--refit',action='store_true')
    args=parser.parse_args(argv)
    export_final(args.audit,args.reuse_audited_model,args.refit)


if __name__=='__main__':main()

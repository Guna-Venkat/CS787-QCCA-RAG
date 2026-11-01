"""Synthetic instrumented control-flow test; real classifier fitting is replaced."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,ClassifierMixin
from ..action_verification_g import controller,protocol


class NonLearningProbe(ClassifierMixin,BaseEstimator):
    """Test double only: constant probabilities, no learned classifier parameters."""
    def fit(self,x,y):
        self.classes_=np.array([0,1]);self.n_features_in_=x.shape[1]
        return self
    def predict_proba(self,x):return np.tile([.6,.4],(len(x),1))


def synthetic_trace():
    cfg=protocol.config('configs/week9/w9g_action_verification.yaml')
    schema=[n for g in cfg['feature_groups'] for n in controller.GROUPS[g]]
    rng=np.random.default_rng(42)
    x=pd.DataFrame(rng.normal(size=(60,32)),columns=schema)
    # Make training-only versus global median/scale treatment observable.
    x.loc[[1,5,13,21,39,50],'next_unselected_cosine']=np.nan
    meta=pd.DataFrame([dict(question_id=str(i),paper_id=str(i//3)) for i in range(60)])
    groups={p:{} for p in ('D0','F0')}
    for i,m in meta.iterrows():
        for p in groups:
            score=.5 if p=='D0' else .8 if i%3==0 else .4
            groups[p][m.question_id]=dict(question_id=m.question_id,paper_id=m.paper_id,
                f1=score,prediction=p,exact_match=0.,final_prompt_tokens=100 if p=='D0' else 200,
                total_query_prompt_tokens=100 if p=='D0' else 300,total_query_generated_tokens=10 if p=='D0' else 20,
                num_lm_calls=1 if p=='D0' else 2,revision_latency=.1)
    _,outer=controller.paper_folds(meta,5)
    expected=[];predict_expected=[];threshold_expected=[]
    for name in cfg['models']:
        for fold,(train,held) in enumerate(outer):
            _,inner=controller.paper_folds(meta.iloc[train].reset_index(drop=True),4)
            for j,(tr,va) in enumerate(inner):
                expected.append((name,set(train[tr]),set(held),'inner',fold,j))
                predict_expected.append(set(train[va]))
            threshold_expected.append(set(meta.iloc[train].question_id))
            expected.append((name,set(train),set(held),'outer',fold,-1));predict_expected.append(set(held))
        threshold_expected.append(set(meta.question_id))
        expected.append((name,set(range(60)),set(),'final',-1,-1))
    original_factory=controller.estimator;original_fit=controller.fit_model
    original_predict=controller.predict_accept;original_threshold=controller.choose_threshold
    fits=[];predictions=[];threshold_calls=[];objects=[];fitted_rows={};recipes=[]

    def factory(name,seed):
        pipeline=original_factory(name,seed)
        recipes.append(dict(model=name,seed=seed,params=pipeline.steps[-1][1].get_params(deep=False)))
        pipeline.steps[-1]=(pipeline.steps[-1][0],NonLearningProbe())
        return pipeline

    def fit(name,seed,x_train,y_train):
        exp=expected[len(fits)];actual=set(x_train.index)
        assert name==exp[0] and actual==exp[1] and not actual&exp[2]
        model=original_fit(name,seed,x_train,y_train)
        # Actual sklearn preprocessors run, but no real logistic/tree learner fits.
        imputer=model.steps[0][1]
        assert np.allclose(imputer.statistics_,x_train.median().to_numpy(),equal_nan=True)
        if name=='logistic':
            scaler=model.steps[1][1];transformed=imputer.transform(x_train)
            assert float(scaler.n_samples_seen_)==len(x_train)
            assert np.allclose(scaler.mean_,transformed.mean(axis=0))
        objects.append(model);fitted_rows[id(model)]=actual
        fits.append(dict(model=name,role=exp[3],outer_fold=exp[4],inner_fold=exp[5],training_rows=len(actual),
            outer_heldout_overlap=len(actual&exp[2]),preprocessing_training_rows_only=True))
        return model

    def predict(model,x_valid):
        actual=set(x_valid.index)
        assert actual==predict_expected[len(predictions)]
        assert not actual&fitted_rows[id(model)]
        predictions.append(dict(validation_rows=len(actual),fit_prediction_row_overlap=0))
        return original_predict(model,x_valid)

    class RestrictedMap(dict):
        def __init__(self,data,allowed):super().__init__(data);self.allowed=allowed;self.read=set()
        def __getitem__(self,key):
            assert key in self.allowed,'Threshold selection read an outer-heldout outcome'
            self.read.add(key);return super().__getitem__(key)

    def threshold(probabilities,all_groups,training_meta,config):
        allowed=threshold_expected[len(threshold_calls)]
        assert set(training_meta.question_id)==allowed
        restricted={p:RestrictedMap(g,allowed) for p,g in all_groups.items()}
        result=original_threshold(probabilities,restricted,training_meta,config)
        reads=set().union(*(g.read for g in restricted.values()))
        assert reads==allowed
        threshold_calls.append(dict(training_queries=len(allowed),outside_allowed_label_reads=0,
            pooled_deployment_selection=len(allowed)==60))
        return result

    def fake_dump(value,path):Path(path).write_bytes(b'SYNTHETIC TEST DOUBLE; NOT A FITTED RESEARCH MODEL')

    with tempfile.TemporaryDirectory(prefix='w9g-audit-trace-') as tmp:
        local=dict(cfg,output_root=tmp)
        with patch.object(controller,'prepare',return_value=(x,meta)),patch.object(controller,'verify_preflight',return_value={}),\
             patch.object(controller,'estimator',side_effect=factory),patch.object(controller,'fit_model',side_effect=fit),\
             patch.object(controller,'predict_accept',side_effect=predict),patch.object(controller,'choose_threshold',side_effect=threshold),\
             patch('joblib.dump',side_effect=fake_dump):
            # Frozen production call graph, synthetic rows, non-learning test doubles.
            controller.train_manually(local,dict(groups=groups,input_hashes={}),dict(synthetic=True),enabled=True)
    assert len(fits)==52 and len(predictions)==50 and len(threshold_calls)==12
    assert len({id(m) for m in objects})==52
    return dict(synthetic_rows=60,synthetic_papers=20,real_classifier_fits=0,
        preprocessing_fits_on_synthetic_training_partitions=52,fresh_pipeline_instances=52,
        validation_prediction_calls=50,inner_and_deployment_threshold_calls=12,
        all_preprocessing_training_only=True,no_prediction_on_fit_rows=True,
        outer_heldout_label_reads_during_inner_threshold_selection=0,
        full_fit_estimators_never_used_for_oof_predictions=True,fits=fits,predictions=predictions,threshold_calls=threshold_calls,
        note='Runtime call-path test only. Original estimator/preprocessor recipes are constructed; classifier steps are replaced by non-learning test doubles. No study controller is retrained.')

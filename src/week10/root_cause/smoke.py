"""Fabricated queries only. No downloaded model, dataset or generator used."""
from .storage import Store,atomic
from .analysis import oracle

def run(path):
    store=Store(path,dict(kind='synthetic',version=1));previous=len(store.rows());calls=0
    def generation(q,k):
        nonlocal calls
        calls+=1;return dict(question_id=q,paper_id='paper'+q,action='k'+str(k),k=k,
            f1=.5 if k==1 else .6,cumulative_prompt_tokens=10*k)
    # Simulate interrupted run: save prefix, then resume twice; generator called once per key.
    for q,k in [('a',1),('a',2)]:
        if store.get([q,k]) is None:store.put([q,k],generation(q,k))
    for _ in range(2):
        for q in ['a','b']:
            for k in [1,2]:
                if store.get([q,k]) is None:store.put([q,k],generation(q,k))
    rows=store.rows();result=dict(queries=2,answers=len(rows),new_fake_calls=calls,no_models_loaded=True,
        exact_oracle=oracle(rows),resume_verified=len(rows)==4 and calls==4-previous)
    if not result['resume_verified']:raise AssertionError('Resume generated a duplicate answer')
    atomic(path/'smoke_result.json',result);return result

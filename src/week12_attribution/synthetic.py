"""Small model-free end-to-end smoke. Never uses real frozen query IDs."""
from src.week12.protocol import Store,digest
from src.week12.algorithm import serialize
from .protocol import directory,study_identity

def answer(e,p):
    from src.week11.evaluation import evaluate
    truth=e['evaluation'].get('gold_label') or e['evaluation']['gold_answer'][0]
    # Synthetic oracle fixture only; real Reader never receives gold labels.
    text=truth if 'red signal' in p['text'] else ('INVALID_OUTPUT' if e['dataset']=='SciFact' else 'unknown')
    return dict(prediction=text,raw_prediction=text,prompt_tokens=p['evidence_tokens']+10,generated_tokens=2,generation_seconds=0.,reader='BASE_FP16',**evaluate(text,e))

def prepare(cfg):
    from src.week11.budgets import FixtureTokenizer
    tok=FixtureTokenizer();f=directory(cfg,True);ident=study_identity(cfg,True);st=Store(f/'prepared',ident);gen=Store(f/'original_fixture_generation',ident)
    for dataset in ['QASPER','HotpotQA','SciFact']:
        text='The red signal was observed. Extra unrelated blue paint.';a=text.index(' Extra');units=[]
        for i,(start,end) in enumerate([(0,a),(a+1,len(text))]):
            units.append(dict(unit_id='synthetic:'+str(i),passage_id='synthetic-p',document_id='synthetic-d',unit_type='SENTENCE',source_text=text[start:end],start_char=start,end_char=end,document_order=[0,0,i],query_relevance=.9-i*.6,parent_retrieval_score=.8,parent_retrieval_rank=1,evidence_token_cost=len(tok.encode(text[start:end])),gold_overlap=i==0))
        e=dict(dataset=dataset,query_id='synthetic-'+dataset,paper_id='synthetic-paper-'+dataset,cluster_id='synthetic-paper-'+dataset,question='Which signal?',task_type='verification' if dataset=='SciFact' else 'qa',split='dev',evaluation=dict(gold_label='SUPPORT') if dataset=='SciFact' else dict(gold_answer=['red signal']))
        for method in ['QBGF_FULL','FIXED_SENTENCE']:
            for b in [256,1024]:
                p=serialize(units,tok);base=answer(e,p);none=answer(e,serialize([],tok));orig=dict(base,evidence_tokens=p['evidence_tokens'],source_hash='SYNTHETIC',evidence_hash=digest(p['evidence_ids']),f1=base.get('f1'),correct=int(base['task_score']==1),gold=e['evaluation'].get('gold_label') or e['evaluation']['gold_answer'])
                st.put([dataset,e['query_id'],method,b],dict(example=e,method=method,budget=b,units=units,full=p,original=orig,none=none,original_evidence_hash=orig['evidence_hash'],source_hash='SYNTHETIC'))
                from .experiment import generation_key
                gen.put(generation_key(e,p),base);gen.put(generation_key(e,serialize([],tok)),none)
    return st.rows(),tok,gen

import pytest
from src.week12.protocol import load
from src.week12_final_review import freeze_final,eligible_methods

def test_no_implicit_freeze():
 assert freeze_final(load(),None,None)['status']=='REVIEW_REQUIRED'

def test_unvalidated_promotions_and_lora_rejected():
 with pytest.raises(ValueError,match='fallback'):freeze_final(load(),'QBGF_FULL','reason')
 with pytest.raises(ValueError,match='LoRA'):freeze_final(load(),'FIXED_SENTENCE','reason',True)

def test_registry_omits_unrun_methods(monkeypatch):
 import src.week12_final_review as m
 monkeypatch.setattr(m,'cohort',lambda cfg:{'D':[dict(query_id='q')]})
 groups={('D',method,b):[dict(query_id='q')] for method in ['NONE','RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL'] for b in [256,1024]}
 groups['D','C2',256]=[dict(query_id='q')]
 pairs,excluded=eligible_methods({'budgets':[256,1024]},groups)
 assert [p[0] for p in pairs]==['NONE','RAW_SKIP','FIXED_SENTENCE','H3','QBGF_FULL']
 assert ['C2','BASE_FP16'] in excluded
 assert not any(p[0]=='LLMLINGUA2' for p in pairs)

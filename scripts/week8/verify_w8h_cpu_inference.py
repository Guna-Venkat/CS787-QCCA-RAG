"""Recompute frozen gate/selector on saved real token diagnostics, without SARA GPU generation.

This is a CPU inference/parity check, not a substitute for the required real GPU smoke.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from src.week8.generation import FrozenRuntimeController,promoted_prompt
from src.week8.generator_features import canonical_k4_context,sha256_file
from src.week8.qcla_experiment import load_inputs,atomic_json
from src.week8.data import load_config
from src.week8.outcome_weighted_training import frozen_policy


def main():
    torch.set_num_threads(4)
    _,reference,_=frozen_policy()
    frozen=json.loads((ROOT/'results/week8/w8e_controller/frozen_controller.json').read_text())
    controller=FrozenRuntimeController(frozen)
    part=load_inputs()['internal_val'];records={str(r['question_id']):r for r in part['records']}
    raw_path=ROOT/'results/week8/w8e_gate_inputs/canonical_diagnostics_internal_val.jsonl'
    provenance=json.loads((raw_path.parent/'provenance_internal_val.json').read_text())
    if sha256_file(raw_path)!=provenance['raw_diagnostics_sha256']:raise ValueError('Raw real-token diagnostic integrity failed')
    raw={str(r['question_id']):r for r in map(json.loads,raw_path.read_text().splitlines())}
    if set(raw)!=set(records):raise ValueError('Real token diagnostics do not cover validation')
    predictions=pd.read_csv(ROOT/'results/week8/w8c_utility/selected_internal_val_predictions.csv',dtype={'question_id':str,'paper_id':str})
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/load_config()['sara']['checkpoint_path']),local_files_only=True)
    cid=tokenizer.convert_tokens_to_ids('<COMPRESS>')
    agreement=0;promotions=0;max_prob_diff=0.;rows=[]
    for q in tqdm(sorted(records),desc='Real CPU gate/selector inference'):
        r=records[q];group=part['routing'][q];emb=part['sfr'][q]
        if raw[q]['draft_answer'].strip()!=str(group.iloc[0].draft_answer).strip() or not raw[q]['greedy_verified']:
            raise ValueError('Raw diagnostic draft changed')
        action,details=controller.decide(r,group,raw[q]['draft_answer'].strip(),raw[q])
        same=action['selected_action']==reference.loc[q,'selected_action']
        if same and action['selected_action']=='PROMOTE':same=int(action['candidate_passage_index'])==int(reference.loc[q,'candidate_passage_index'])
        agreement+=int(same)
        current=pd.DataFrame(details['candidate_predictions'])
        merged=current.merge(predictions,on=['question_id','paper_id','semantic_rank','candidate_passage_index'],suffixes=('_new','_saved'),validate='one_to_one')
        if len(merged)!=6:raise ValueError('Candidate probability coverage differs')
        error=max(float(np.abs(merged[k+'_new']-merged[k+'_saved']).max()) for k in ['prob_help','prob_hurt'])
        max_prob_diff=max(max_prob_diff,error)
        _,base_prompt,_,compressed,_,_=canonical_k4_context(group,r,emb,tokenizer)
        assert tokenizer(base_prompt)['input_ids'].count(cid)==len(compressed)==6
        if action['selected_action']=='PROMOTE':
            promotions+=1
            a=promoted_prompt(r,group,emb,tokenizer,action,'G0',raw[q]['draft_answer'])
            b=promoted_prompt(r,group,emb,tokenizer,action,'G1',raw[q]['draft_answer'])
            assert a[2:]==b[2:] and torch.equal(a[1],b[1])
            assert len(a[2])==5 and len(a[3])==5 and set(a[2])|set(a[3])==set(range(10))
            assert not (set(a[2])&set(a[3]))
            assert tokenizer(a[0])['input_ids'].count(cid)==tokenizer(b[0])['input_ids'].count(cid)==5
        rows.append({'question_id':q,'action_match':same,'action':action['selected_action'],'probability_max_abs_diff':error})
    result={'queries':len(records),'action_matches_saved_E5':agreement,'promotions':promotions,
        'U4A_probability_max_abs_diff':max_prob_diff,'all_G0_G1_prompt_slot_checks_passed':True,
        'mode':'actual CPU model inference using saved real original-SARA token diagnostics; no new SARA generation',
        'GPU_smoke_completed':False,'cuda_available_in_agent_session':torch.cuda.is_available(),
        'rows':rows}
    out=ROOT/'results/week8/w8h_full_system';out.mkdir(exist_ok=True,parents=True)
    atomic_json(out/'cpu_inference_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if agreement!=len(records) or max_prob_diff>1e-4:raise ValueError('Runtime inference disagrees with frozen reference; inspect verification report')

if __name__=='__main__':main()

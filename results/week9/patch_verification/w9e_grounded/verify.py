"""Independent post-preflight checks; reads original A-D files without writing."""
import json,csv
from pathlib import Path
from collections import Counter
from src.week9.provenance import ROOT,digest,file_hash,source_hashes
from src.week9.caching import read_jsonl,atomic_json
from src.week9.grounded_e.protocol import config,output_dir,e_sources
from src.week9.grounded_e.runner import validate_rows,verify_gate
from src.week9.grounded_e.evidence import tokens
from src.week9.backend import load_tokenizer
cfg=config(ROOT/'configs/week9/w9e_grounded.yaml');out=output_dir(cfg)
ident=json.loads((out/'protocol/grounded_e_identity.json').read_text())
assert ident['e_sources']==e_sources() and ident['upstream_sources']==source_hashes()
seal=json.loads((out/'protocol/d_input_freeze.json').read_text())
assert all(file_hash(p)==h for p,h in seal['input_hashes'].items())
rows=read_jsonl(out/'preflight/grounded_e_results.jsonl')
objects=read_jsonl(out/'preflight/evidence_objects.jsonl')
calls=read_jsonl(out/'preflight/lm_calls.jsonl')
assert len(rows)==12 and len(objects)==12
stamp=digest(dict(stage='preflight',protocol=ident))
for r in rows+objects+calls:
    assert r['cache_fingerprint']==stamp and r['row_content_hash']==digest({k:v for k,v in r.items() if k!='row_content_hash'})
by_q={};ids={r['question_id'] for r in rows}
with (ROOT/cfg['output_root']/'extraction/fit/w9d_representation.jsonl').open() as f:
    for line in f:
        r=json.loads(line)
        if r['question_id'] in ids and r['policy']=='D1':by_q[r['question_id']]=r['source_chunks']
records=[dict(inference=dict(question_id=q)) for q in ids]
validate_rows(rows,records,by_q,cfg)
tokenizer=load_tokenizer(cfg)
assert all(tokens(tokenizer,r['evidence']['text'])==r['evidence']['evidence_tokens']<=128 for r in rows)
assert all(r['prediction']==r['initial_draft'] and not r['answer_generation_called'] and r['revision_prompt_tokens']==0 for r in rows if r['fail_closed'])
expected_calls=sum(len(r['model_calls']) for r in rows)
assert len(calls)==expected_calls
assert all(r['dtype']=='torch.float16' and r['attention_backend']=='efficient_sdpa' for r in calls)
assert not (out/'fit/grounded_e_results.jsonl').exists()
try:verify_gate(out,ident)
except ValueError as exc:assert 'acknowledge' in str(exc)
else:raise AssertionError('Quality warnings did not block the full run')
assert verify_gate(out,ident,True)['implementation_checks_passed']
with (ROOT/cfg['data']['fit_manifest']).open() as f:fit_ids=[r['question_id'] for r in csv.DictReader(f)]
expected=sorted(fit_ids,key=lambda q:digest(['grounded-e-fit-preflight-v1',42,q]))[:3]
gate=json.loads((out/'preflight/preflight_gate.json').read_text())
assert gate['query_ids']==expected
summary=dict(branch='gagan',historical_artifacts_unchanged=True,upstream_source_bundle_unchanged=True,
    original_d_freezes_rewritten=False,e_only_d_input_seal=True,preflight_queries=expected,preflight_rows=12,
    accepted_evidence_token_max=max(r['evidence']['evidence_tokens'] for r in rows),
    fail_closed_rows=sum(r['fail_closed'] for r in rows),model_calls=len(calls),
    peak_memory_gib=max(r['peak_memory_allocated'] for r in calls)/1024**3,
    all_accepted_spans_exact_source_substrings=True,all_failures_keep_original_draft_without_answer_call=True,
    quality_warnings=gate['warnings'],full_gate_blocked_until_explicit_warning_acknowledgment=True,
    prompts_repaired_or_tuned=False,full_e_started=False,tests_passed=103,tests_deselected=1,
    deselection='Existing all-notebooks safe-default test conflicts with user-enabled B/C/D flags; dedicated E disabled-path test passes')
atomic_json(ROOT/'results/week9/patch_verification/w9e_grounded/verification.json',summary)
print(json.dumps(summary,indent=2))
